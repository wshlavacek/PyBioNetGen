import re
from typing import NoReturn

from bionetgen.core.exc import BNGParseError
from bionetgen.core.utils.logging import BNGLogger

from .blocks import (
    CompartmentBlock,
    EnergyPatternBlock,
    FunctionBlock,
    MoleculeTypeBlock,
    ObservableBlock,
    ParameterBlock,
    PopulationMapBlock,
    RuleBlock,
    SpeciesBlock,
)
from .pattern import Component, Molecule, Pattern
from .rulemod import RuleMod

logger = BNGLogger()
_PARSE_SOURCE = "<BNG-XML>"


def _raise_parse_error(message: str, *, loc: str) -> NoReturn:
    logger.error(message, loc=loc)
    raise BNGParseError(_PARSE_SOURCE, message=f": {message}")


# BNG2.pl serializes the boolean operators ``&&`` and ``||`` in <Expression>
# elements as the function-call-shaped strings ``(operand)and(operand)`` and
# ``(operand)or(operand)`` (see Perl2/Expression.pm: '&&' => 'and'). When we
# re-emit that body into a .bngl file BNG2.pl re-parses ``and(...)`` as a
# function call and aborts ("Missing end parentheses ... at and(...)"), so
# we have to undo the substitution here. The match is intentionally
# anchored on the closing-paren of the left operand: BNG2.pl always wraps
# both operands in parens when emitting these forms, so ``)and(`` /
# ``)or(`` only appear as the boolean-operator encoding — never as the
# tail of a user-defined identifier or a literal function call.
_AND_OP_RE = re.compile(r"\)and\(")
_OR_OP_RE = re.compile(r"\)or\(")


def _decode_xml_boolean_ops(expr: str) -> str:
    """Translate BNG2.pl's ``)and(`` / ``)or(`` XML encoding back to ``&&`` / ``||``."""
    if not expr:
        return expr
    expr = _AND_OP_RE.sub(") && (", expr)
    expr = _OR_OP_RE.sub(") || (", expr)
    return expr


def _ratelaw_arg_ids(args_xml) -> str:
    """Join the ``@id`` of each Argument in a ListOfArguments[N] element.

    BNG-XML packs a single Argument as a dict and multiple as a list, so
    we accept both shapes. Returns "" when ``args_xml`` is None / empty
    so callers can render zero-arg ``f()`` consistently.
    """
    if not args_xml:
        return ""
    args = args_xml.get("Argument") if hasattr(args_xml, "get") else None
    if args is None:
        return ""
    if isinstance(args, list):
        return ",".join(str(a["@id"]) for a in args)
    return str(args["@id"])


def _resolve_ratelaw(xml, *, context: str, loc: str) -> str:
    rate_type = str(xml["@type"])
    if rate_type == "Ele":
        rate_cts_xml = xml["ListOfRateConstants"]
        return str(rate_cts_xml["RateConstant"]["@value"])
    if rate_type == "Function":
        return str(xml["@name"])
    if rate_type == "FunctionProduct":
        # Mirror BNG2.pl/Perl2/RateLaw.pm:670-677 — emit
        # FunctionProduct("name1(args1)","name2(args2)") so the BNGL
        # round-trips through BNG2.pl's parser and reaches NFsim, which
        # supports FunctionProduct natively (NFinput.cpp:2251).
        name1 = str(xml["@name1"])
        name2 = str(xml["@name2"])
        a1 = _ratelaw_arg_ids(xml.get("ListOfArguments1"))
        a2 = _ratelaw_arg_ids(xml.get("ListOfArguments2"))
        return f'FunctionProduct("{name1}({a1})","{name2}({a2})")'
    if rate_type in ("MM", "Sat", "Hill", "Arrhenius"):
        # A function type
        rate_cts = rate_type + "("
        args = xml["ListOfRateConstants"]["RateConstant"]
        if isinstance(args, list):
            for iarg, arg in enumerate(args):
                if iarg > 0:
                    rate_cts += ","
                rate_cts += str(arg["@value"])
        else:
            rate_cts += str(args["@value"])
        rate_cts += ")"
        return rate_cts
    _raise_parse_error(
        f"Unrecognized rate law type {rate_type!r} in {context}",
        loc=loc,
    )


###### Base object  ######
class XMLObj:
    """
    Base object that contains XMLs that is the parent of
    all XML object that will be used for BNGL blocks as
    we read from BNGXML. This sets up the python internals
    such as __repr__ and __str__ for the rest of the XML
    classes.

    This base class assumes at least two methods will be
    defined by each class that inherits this base object
    which are defined in the methods below.

    Attributes
    ----------
    xml : list/OrderedDict
        XML loaded via xmltodict to be parsed. Either a
        list of items or an OrderedDict.
    parsed_obj : obj
        appropriate parsed object, one of the Block objects

    Methods
    -------
    resolve_xml(xml)
        the method that parses the XML given and is written
        separately for each subclass of this one
    gen_string()
        the method to generate the string from the parsed
        object
    """

    def __init__(self, xml):
        self.xml = xml
        self.parsed_obj = self.parse_xml(self.xml)

    def __repr__(self):
        return self.gen_string()

    def __str__(self):
        return self.gen_string()

    def gen_string(self):
        return str(self.parsed_obj)

    def parse_xml(self, xml):
        """ """
        raise NotImplementedError


###### Fundamental parsing objects ######
# This is for handling bond XMLs
class BondsXML:
    """
    Bonds XML parser, derived from XMLObj.
    """

    def __init__(self, bonds_xml=None):
        self.bonds_dict = {}
        if bonds_xml is not None:
            self.resolve_xml(bonds_xml)

    def set_xml(self, bonds_xml):
        self.resolve_xml(bonds_xml)

    def get_bond_id(self, comp):
        # Get the ID of the bond from an XML id something
        # belongs to, e.g. O1_P1_M1_C2
        num_bonds = comp["@numberOfBonds"]
        comp_id = comp["@id"]
        try:
            num_bonds = int(num_bonds)
        except (TypeError, ValueError):
            # This means we have something like +/?
            return num_bonds
        # use the comp_id to find the bond index from
        # self.bonds_dict
        comp_key = self.get_tpl_from_id(comp_id)
        bond_id = self.bonds_dict[comp_key]
        return bond_id

    def get_tpl_from_id(self, id_str):
        # ID str is looking like O1_P1_M2_C3
        # we are going to assume a 4-tuple per key
        id_list = id_str.split("_")
        id_tpl = tuple(id_list)
        return id_tpl

    def tpls_from_bond(self, bond):
        s1 = bond["@site1"]
        s2 = bond["@site2"]
        id_list_1 = s1.split("_")
        s1_tpl = tuple(id_list_1)
        id_list_2 = s2.split("_")
        s2_tpl = tuple(id_list_2)
        return (s1_tpl, s2_tpl)

    def resolve_xml(self, bonds_xml):
        # self.bonds_dict is a dictionary you can key
        # with the tuple taken from the ID and then
        # get a bond ID cleanly
        if isinstance(bonds_xml, list):
            for ibond, bond in enumerate(bonds_xml):
                bond_partner_1, bond_partner_2 = self.tpls_from_bond(bond)
                if bond_partner_1 not in self.bonds_dict:
                    self.bonds_dict[bond_partner_1] = [ibond + 1]
                else:
                    # B13: must append the int, not a [int] list — the
                    # rendered Component.__str__ emits ``f"!{bond}"`` per
                    # entry, so a list ``[2]`` would render as ``![2]``,
                    # which BNG2.pl reads as a compartment specifier.
                    self.bonds_dict[bond_partner_1].append(ibond + 1)
                if bond_partner_2 not in self.bonds_dict:
                    self.bonds_dict[bond_partner_2] = [ibond + 1]
                else:
                    self.bonds_dict[bond_partner_2].append(ibond + 1)
        else:
            bond_partner_1, bond_partner_2 = self.tpls_from_bond(bonds_xml)
            self.bonds_dict[bond_partner_1] = [1]
            self.bonds_dict[bond_partner_2] = [1]


# The full pattern parser
class PatternXML(XMLObj):
    """
    Pattern XML parser, derived from XMLObj.

    Methods
    -------
    _process_mol(xml)
        processes a molecule XML dictionary and returns
        a Molecule object
    _process_comp(self, xml)
        processes a component XML dictionary and returns
        a list of Compartment objects
    """

    def __init__(self, xml) -> None:
        super().__init__(xml)

    def parse_xml(self, xml) -> Pattern:
        # initialize
        pattern = Pattern()
        if "ListOfBonds" in xml:
            bonds = BondsXML(xml["ListOfBonds"]["Bond"])
            pattern._bonds = bonds
            self._bonds = bonds
        else:
            bonds = BondsXML()
            pattern._bonds = bonds
            self._bonds = bonds
        # check for compartment and add if exists
        if "@compartment" in xml:
            pattern.compartment = xml["@compartment"]
        # check for label and add if exists
        if "@label" in xml:
            pattern.label = xml["@label"]
        # check to see if pattern is fixed
        if "@Fixed" in xml:
            if xml["@Fixed"] == "1":
                pattern.fixed = True
        # check to see if pattern is only matched once
        if "@matchOnce" in xml:
            if xml["@matchOnce"] == "1":
                pattern.MatchOnce = True
        # check for relation & quantity, add if exist
        if ("@relation" in xml) and ("@quantity" in xml):
            relation = xml["@relation"]
            quantity = xml["@quantity"]
            pattern.relation = relation
            try:
                n = int(quantity)
                f = float(quantity)
            except (TypeError, ValueError):
                _raise_parse_error(
                    f"Pattern quantity must be an integer, got {quantity!r}",
                    loc=f"{__file__} : PatternXML.parse_xml()",
                )
            if n != f:
                _raise_parse_error(
                    f"Pattern quantity must be an integer, got {quantity!r}",
                    loc=f"{__file__} : PatternXML.parse_xml()",
                )
            pattern.quantity = quantity
        # check for either list of molecules or single molecule, add if exist
        mols = xml["ListOfMolecules"]["Molecule"]
        molecules = []
        if isinstance(mols, list):
            # list of molecules
            for imol, mol in enumerate(mols):
                mol_obj = self._process_mol(mol)
                molecules.append(mol_obj)
        else:
            # a single molecule
            mol_obj = self._process_mol(mols)
            molecules.append(mol_obj)
        pattern.molecules = molecules
        # return a complete pattern object
        return pattern

    def _process_mol(self, xml) -> Molecule:
        """ """
        # initialize
        molecule = Molecule()
        #
        if "@name" in xml:
            molecule.name = xml["@name"]
        #
        if "@label" in xml:
            molecule.label = xml["@label"]
        #
        if "@compartment" in xml:
            molecule.compartment = xml["@compartment"]
        #
        if "ListOfComponents" in xml:
            # Single molecule can't have bonds
            molecule.components = self._process_comp(
                xml["ListOfComponents"]["Component"]
            )
        #
        return molecule

    def _process_comp(self, xml) -> list:
        # bonds = compartment id, bond id
        # comp xml can be a list or a dict
        comp_list = []
        if isinstance(xml, list):
            # we have multiple and this is a list
            for icomp, comp in enumerate(xml):
                component = Component()
                component.name = comp["@name"]
                if "@label" in comp:
                    component.label = comp["@label"]
                if "@state" in comp:
                    component.state = comp["@state"]
                if comp["@numberOfBonds"] != "0":
                    # DEBUG
                    bond_id = self._bonds.get_bond_id(comp)
                    for bi in bond_id:
                        component.bonds.append(bi)
                comp_list.append(component)
        else:
            # single comp, this is a dict
            component = Component()
            component.name = xml["@name"]
            if "@label" in xml:
                component.label = xml["@label"]
            if "@state" in xml:
                component.state = xml["@state"]
            if xml["@numberOfBonds"] != "0":
                bond_id = self._bonds.get_bond_id(xml)
                for bi in bond_id:
                    component.bonds.append(bi)
            comp_list.append(component)
        return comp_list


# Helper parser for pattern list (for observables)
class PatternListXML:
    """
    Pattern list XML parser. Takes a list of patterns
    and parses them using PatternXML and generates a
    list of patterns.

    Attributes
    ----------
    patterns : list[Pattern]
        list of patterns parsed from the XML dict list
    """

    def __init__(self, xml) -> None:
        self.patterns = self.parse_xml(xml)

    def parse_xml(self, xml) -> list:
        pats = xml["Pattern"]
        patterns = []
        if isinstance(pats, list):
            # we have multiple patterns so this is a list
            for ipattern, pattern in enumerate(pats):
                patterns.append(PatternXML(pattern).parsed_obj)
        else:
            patterns.append(PatternXML(pats).parsed_obj)
        return patterns


###### Parsers ######
class ParameterBlockXML(XMLObj):
    """
    PatternBlock XML parser, derived from XMLObj. Creates
    a ParameterBlock that contains the parameters parsed.
    """

    def __init__(self, xml) -> None:
        super().__init__(xml)

    def parse_xml(self, xml) -> ParameterBlock:
        # make block
        block = ParameterBlock()
        # parse parameters
        if isinstance(xml, list):
            for b in xml:
                # add content to line
                name = b["@id"]
                value = b["@value"]
                # If "@expr" is set, it supercedes value
                if "@expr" in b:
                    value = b["@expr"]
                block.add_parameter(name, value)
        else:
            # add content to line
            name = xml["@id"]
            value = xml["@value"]
            if "@expr" in xml:
                value = xml["@expr"]
            # add to list of lines
            block.add_parameter(name, value)
        block.reset_compilation_tags()
        return block


#
class CompartmentBlockXML(XMLObj):
    """
    CompartmentBlock XML parser, derived from XMLObj. Creates
    a CompartmentBlock that contains the compartments parsed.
    """

    def __init__(self, xml) -> None:
        super().__init__(xml)

    def parse_xml(self, xml):
        block = CompartmentBlock()

        if isinstance(xml, list):
            for comp in xml:
                cname = comp["@id"]
                dim = comp["@spatialDimensions"]
                size = comp["@size"]
                outside = None
                if "@outside" in comp:
                    outside = comp["@outside"]
                block.add_compartment(cname, dim, size, outside=outside)
        else:
            cname = xml["@id"]
            dim = xml["@spatialDimensions"]
            size = xml["@size"]
            outside = None
            if "@outside" in xml:
                outside = xml["@outside"]
            block.add_compartment(cname, dim, size, outside=outside)

        block.reset_compilation_tags()
        return block


#
class ObservableBlockXML(XMLObj):
    """
    ObservableBlock XML parser, derived from XMLObj. Creates
    an ObservableBlock that contains the observables parsed.
    """

    def __init__(self, xml) -> None:
        super().__init__(xml)

    def parse_xml(self, xml) -> ObservableBlock:
        #
        block = ObservableBlock()
        #
        if isinstance(xml, list):
            for b in xml:
                name = b["@name"]
                otype = b["@type"]
                patterns = PatternListXML(b["ListOfPatterns"]).patterns
                block.add_observable(name, otype, patterns)
        else:
            name = xml["@name"]
            otype = xml["@type"]
            patterns = PatternListXML(xml["ListOfPatterns"]).patterns
            block.add_observable(name, otype, patterns)
        #
        return block


#
class SpeciesBlockXML(XMLObj):
    """
    SpeciesBlock XML parser, derived from XMLObj. Creates
    a SpeciesBlock that contains the species parsed.
    """

    def __init__(self, xml):
        super().__init__(xml)

    def parse_xml(self, xml):
        block = SpeciesBlock()

        if isinstance(xml, list):
            # we have multiple patterns so this is a list
            for ispec, spec in enumerate(xml):
                pattern = PatternXML(spec)
                count = spec["@concentration"]
                block.add_species(pattern, count)
        else:
            pattern = PatternXML(xml)
            count = xml["@concentration"]
            block.add_species(pattern, count)

        return block


#
class MoleculeTypeBlockXML(XMLObj):
    """
    MoleculeTypeBlock XML parser, derived from XMLObj. Creates
    a MoleculeTypeBlock that contains the molecule types parsed.

    Methods
    -------
    add_molecule_type_to_block(block,xml)
        parses the XML to create a MoleculeType object and
        adds it to a given MoleculeTypeBlock object
    """

    def __init__(self, xml):
        super().__init__(xml)

    def parse_xml(self, xml):
        block = MoleculeTypeBlock()
        #
        if isinstance(xml, list):
            for md in xml:
                self.add_molecule_type_to_block(block, md)
        else:
            self.add_molecule_type_to_block(block, xml)
        #
        return block

    def add_molecule_type_to_block(self, block, xml):
        name = xml["@id"]
        components = []
        if "ListOfComponentTypes" in xml:
            comp_obj = Component()
            comp_dict = xml["ListOfComponentTypes"]["ComponentType"]
            if "@id" in comp_dict:
                comp_obj.name = comp_dict["@id"]
                if "ListOfAllowedStates" in comp_dict:
                    # we have states
                    al_states = comp_dict["ListOfAllowedStates"]["AllowedState"]
                    if isinstance(al_states, list):
                        for istate, state in enumerate(al_states):
                            comp_obj.states.append(state["@id"])
                    else:
                        comp_obj.states.append(al_states["@id"])
                components.append(comp_obj)
            else:
                # multiple components
                for icomp, comp in enumerate(comp_dict):
                    comp_obj = Component()
                    comp_obj.name = comp["@id"]
                    if "ListOfAllowedStates" in comp:
                        # we have states
                        al_states = comp["ListOfAllowedStates"]["AllowedState"]
                        if isinstance(al_states, list):
                            for istate, state in enumerate(al_states):
                                comp_obj.states.append(state["@id"])
                        else:
                            comp_obj.states.append(al_states["@id"])
                    components.append(comp_obj)
        block.add_molecule_type(name, components)


class FunctionBlockXML(XMLObj):
    """
    FunctionBlock XML parser, derived from XMLObj. Creates
    a FunctionBlock that contains the functions parsed.

    Methods
    -------
    get_arguments(xml)
        parses the argument list XML and returns a list of
        argument names
    """

    def __init__(self, xml):
        super().__init__(xml)

    def parse_xml(self, xml):
        block = FunctionBlock()
        #
        if isinstance(xml, list):
            for f in xml:
                fname = f["@id"]
                expr = self._resolve_expression(f)
                args = []
                if "ListOfArguments" in f:
                    args = self.get_arguments(f["ListOfArguments"]["Argument"])
                block.add_function(fname, expr, args=args)
        else:
            fname = xml["@id"]
            expr = self._resolve_expression(xml)
            args = []
            if "ListOfArguments" in xml:
                args = self.get_arguments(xml["ListOfArguments"]["Argument"])
            block.add_function(fname, expr, args=args)

        return block

    def _resolve_expression(self, f) -> str:
        """Return the BNGL expression body for a Function XML element.

        Most functions serialize their body verbatim into ``<Expression>``,
        but BNG2.pl rewrites ``tfun(...)`` calls (both the inline-array form
        and the file-based ``TFUN(arg, "file")`` form) as the placeholder
        ``__TFUN_VAL__`` and stashes the real arguments in attributes on
        the ``<Function>`` element. Round-tripping the placeholder back into
        BNGL is invalid — BNG2.pl can't re-parse it. Reconstruct the call
        from the attributes when present.
        """
        raw = str(f.get("Expression", ""))
        if f.get("@type") != "TFUN":
            return _decode_xml_boolean_ops(raw)

        ctr = str(f.get("@ctrName", ""))
        if "@xData" in f:
            xs = str(f.get("@xData", ""))
            ys = str(f.get("@yData", ""))
            method = str(f.get("@method", "linear"))
            body = f"tfun([{xs}],[{ys}],{ctr}"
            if method and method != "linear":
                body += f',method=>"{method}"'
            body += ")"
        elif "@file" in f:
            body = f'TFUN({ctr},"{str(f.get("@file", ""))}")'
        else:
            body = raw

        for placeholder in ("__TFUN__VAL__", "__TFUN_VAL__"):
            if placeholder in raw:
                return _decode_xml_boolean_ops(raw.replace(placeholder, body).strip())
        return _decode_xml_boolean_ops(body.strip())

    def get_arguments(self, xml) -> list:
        if isinstance(xml, list):
            return [arg["@id"] for arg in xml]
        else:
            return [xml["@id"]]


class RuleBlockXML(XMLObj):
    """
    RuleBlock XML parser, derived from XMLObj. Creates
    a RuleBlock that contains the rules parsed.

    Methods
    -------
    resolve_ratelaw(xml)
        parses a rate law XML and returns the rate constant
    resolve_rxn_side(xml)
        parses either reactants or products XML and returns
        a list of Pattern objects
    get_operations(xml)
        creates and populates a list of operations and their arguments
    get_rule_mod(xml)
        stores rule modifier used by a given rule as a string
    """

    def __init__(self, xml):
        self.bidirectional = False
        super().__init__(xml)

    def parse_xml(self, xml):
        block = RuleBlock()

        # check for multiple rules and parse each one
        if isinstance(xml, list):
            for irule, rule in enumerate(xml):
                name = rule["@name"]
                reactants = self.resolve_rxn_side(rule["ListOfReactantPatterns"])
                products = self.resolve_rxn_side(rule["ListOfProductPatterns"])
                if "RateLaw" not in rule:
                    _raise_parse_error(
                        f"Reaction rule {name!r} is missing a RateLaw entry",
                        loc=f"{__file__} : RuleBlockXML.parse_xml()",
                )
                rate_constants = [self.resolve_ratelaw(rule["RateLaw"])]
                rule_modifier = self.get_rule_mod(rule)
                operations = []
                if rule["ListOfOperations"] is not None:
                    if len(rule["ListOfOperations"]) > 0:
                        operations = self.get_operations(rule["ListOfOperations"])

                block.add_rule(
                    name,
                    reactants=reactants,
                    products=products,
                    rate_constants=rate_constants,
                    rule_mod=rule_modifier,
                    operations=operations,
                )
        else:
            name = xml["@name"]
            reactants = self.resolve_rxn_side(xml["ListOfReactantPatterns"])
            products = self.resolve_rxn_side(xml["ListOfProductPatterns"])
            if "RateLaw" not in xml:
                _raise_parse_error(
                    f"Reaction rule {name!r} is missing a RateLaw entry",
                    loc=f"{__file__} : RuleBlockXML.parse_xml()",
                )
            rate_constants = [self.resolve_ratelaw(xml["RateLaw"])]
            rule_modifier = self.get_rule_mod(xml)
            if xml["ListOfOperations"] is None:
                operations = []
            else:
                operations = self.get_operations(xml["ListOfOperations"])

            block.add_rule(
                name,
                reactants=reactants,
                products=products,
                rate_constants=rate_constants,
                rule_mod=rule_modifier,
                operations=operations,
            )
        block.consolidate_rules()
        return block

    def resolve_ratelaw(self, xml):
        return _resolve_ratelaw(
            xml,
            context="reaction rule",
            loc=f"{__file__} : RuleBlockXML.resolve_ratelaw()",
        )

    def resolve_rxn_side(self, xml):
        # this is either reactant or product
        if xml is None:
            return [Molecule()]
        elif "ReactantPattern" in xml:
            # this is a lhs/reactant side
            sl = []
            side = xml["ReactantPattern"]
            if "@compartment" in side:
                self.react_comp = side["@compartment"]
            else:
                self.react_comp = None
            if isinstance(side, list):
                # this is a list of reactant patterns
                for ireact, react in enumerate(side):
                    sl.append(PatternXML(react).parsed_obj)
            else:
                sl.append(PatternXML(side).parsed_obj)
            return sl
        elif "ProductPattern" in xml:
            # rhs/product side
            side = xml["ProductPattern"]
            sl = []
            if "@compartment" in side:
                self.prod_comp = side["@compartment"]
            else:
                self.prod_comp = None
            if isinstance(side, list):
                # this is a list of product patterns
                for iprod, prod in enumerate(side):
                    sl.append(PatternXML(prod).parsed_obj)
            else:
                sl.append(PatternXML(side).parsed_obj)
            return sl
        else:
            _raise_parse_error(
                "Reaction side XML must contain ReactantPattern or ProductPattern",
                loc=f"{__file__} : RuleBlockXML.resolve_rxn_side()",
            )

    def get_operations(self, xml):
        ops = []
        # List all possible operations & arguments
        ops_types = [
            "AddBond",
            "DeleteBond",
            "ChangeCompartment",
            "StateChange",
            "Add",
            "Delete",
        ]
        op_args = [
            "@site1",
            "@site2",
            "@id",
            "@source",
            "@destination",
            "@flipOrientation",
            "@moveConnected",
            "@site",
            "@finalState",
            "@DeleteMolecules",
        ]
        # Loop through valid arguments and record
        for op_type in ops_types:
            if op_type in xml:
                for op_arg in op_args:
                    if op_arg in xml:
                        n_op = xml[op_arg]
                        ops.append(n_op)
        return ops

    def get_rule_mod(self, xml):
        rule_mod = RuleMod()
        list_ops = xml.get("ListOfOperations")
        had_explicit_ops = list_ops is not None
        if list_ops is None:
            list_ops = {}
        # determine which rule mod is being used, if any
        if "Delete" in list_ops:
            del_op = list_ops["Delete"]
            if not isinstance(del_op, list):
                del_op = [del_op]  # Make sure del_op is list
            dmvals = [op.get("@DeleteMolecules", "0") for op in del_op]
            # All Delete operations in rule must have DeleteMolecules attribute or
            # it does not apply to the whole rule
            if all(str(val) == "1" for val in dmvals):
                rule_mod.type = "DeleteMolecules"
                rule_mod.add_modifier("DeleteMolecules")
                # JRF: I don't believe the id of the specific op rule_mod is currently used
                # rule_mod.id = op["@id"]
        elif "ChangeCompartment" in list_ops:
            move_op = list_ops["ChangeCompartment"]
            if not isinstance(move_op, list):
                # get mod information & add to string
                mod_call = move_op["@moveConnected"]
                # check if modifier was called or automatic
                if mod_call == "1":
                    rule_mod.type = "MoveConnected"
                    rule_mod.add_modifier("MoveConnected")
                    rule_mod.id = move_op["@id"]
                    rule_mod.source = move_op["@source"]
                    rule_mod.destination = move_op["@destination"]
                    rule_mod.flip = move_op["@flipOrientation"]
                    rule_mod.call = mod_call
            else:
                rule_mod.id = []
                rule_mod.source = []
                rule_mod.destination = []
                rule_mod.flip = []
                rule_mod.call = []
                for mo in move_op:
                    if mo["@moveConnected"] == "1":
                        rule_mod.type = "MoveConnected"
                        rule_mod.add_modifier("MoveConnected")
                        rule_mod.id.append(mo["@id"])
                        rule_mod.source.append(mo["@source"])
                        rule_mod.destination.append(mo["@destination"])
                        rule_mod.flip.append(mo["@flipOrientation"])
                        rule_mod.call.append(mo["@moveConnected"])
        elif "RateLaw" in xml:
            # check if modifier is called
            ratelaw = xml["RateLaw"]
            rate_type = ratelaw["@type"]
            if rate_type == "Function" and str(ratelaw.get("@totalrate", "0")) == "1":
                rule_mod.type = "TotalRate"
                rule_mod.add_modifier("TotalRate")
                rule_mod.id = ratelaw["@id"]
                rule_mod.rate_type = ratelaw["@type"]
                rule_mod.name = ratelaw["@name"]
                rule_mod.call = ratelaw.get("@totalrate", "0")

        for key, value in xml.items():
            if key not in (
                "ListOfIncludeReactants",
                "ListOfIncludeProducts",
                "ListOfExcludeReactants",
                "ListOfExcludeProducts",
            ):
                continue
            selectors = value if isinstance(value, list) else [value]
            for selector in selectors:
                call = self._build_selector_modifier(key, selector)
                if call is not None:
                    rule_mod.add_modifier(call)

        if (
            rule_mod.type is None
            and len(rule_mod.modifiers) == 0
            and not had_explicit_ops
        ):
            return None
        return rule_mod

    def _build_selector_modifier(self, key, selector_xml):
        call_names = {
            "ListOfIncludeReactants": "include_reactants",
            "ListOfExcludeReactants": "exclude_reactants",
            "ListOfIncludeProducts": "include_products",
            "ListOfExcludeProducts": "exclude_products",
        }
        call_name = call_names.get(key)
        if call_name is None:
            return None
        selector_id = str(selector_xml.get("@id", ""))
        match = re.search(r"_(?:RP|PP)(\d+)$", selector_id)
        if match is None:
            return None
        pattern_xml = selector_xml.get("Pattern")
        if pattern_xml is None:
            return None
        patterns = pattern_xml if isinstance(pattern_xml, list) else [pattern_xml]
        pattern_parts = [self._format_selector_pattern(pattern) for pattern in patterns]
        pattern_str = " + ".join(pattern_parts)
        return f"{call_name}({match.group(1)},{pattern_str})"

    def _format_selector_pattern(self, pattern_xml):
        pattern = PatternXML(pattern_xml).parsed_obj
        if len(pattern.molecules) == 1:
            molecule = pattern.molecules[0]
            if (
                molecule.name != "0"
                and len(molecule.components) == 0
                and molecule.compartment is None
                and molecule.label is None
                and pattern.compartment is None
                and not pattern.fixed
                and not pattern.MatchOnce
                and pattern.relation is None
                and pattern.quantity is None
            ):
                return molecule.name
        return str(pattern)


class EnergyPatternBlockXML(XMLObj):
    """
    EnergyPatternBlock XML parser, derived from XMLObj. Creates
    an EnergyPatternBlock that contains the energy patterns parsed.
    """

    def __init__(self, xml):
        super().__init__(xml)

    def parse_xml(self, xml):
        block = EnergyPatternBlock()

        if isinstance(xml, list):
            for b in xml:
                # get id & expression
                epid = b["@id"]
                expr = b["@expression"]
                # get pattern
                pattern_node = b["Pattern"]
                pattern = PatternXML(pattern_node).parsed_obj
                block.add_energy_pattern(epid, pattern, expr)
        else:
            # get id & expression
            epid = xml["@id"]
            expr = xml["@expression"]
            # get pattern
            pattern_node = xml["Pattern"]
            pattern = PatternXML(pattern_node).parsed_obj
            block.add_energy_pattern(epid, pattern, expr)

        return block


class PopulationMapBlockXML(XMLObj):
    """
    PopulationMapBlock XML parser, derived from XMLObj. Creates
    a PopulationMapBlock that contains the population maps parsed.

    Methods
    -------
    resolve_ratelaw(xml)
        parses a rate law XML and returns the rate constant
    """

    def __init__(self, xml):
        super().__init__(xml)

    def parse_xml(self, xml):
        block = PopulationMapBlock()

        if isinstance(xml, list):
            for b in xml:
                # get id
                pmid = b["@id"]
                # get structured species
                struct_spec_node = b["StructuredSpecies"]
                spec_node = struct_spec_node["Species"]
                struct_spec = PatternXML(spec_node).parsed_obj
                # get population species
                pop_spec_node = b["PopulationSpecies"]
                pop_node = pop_spec_node["Species"]
                pop_spec = PatternXML(pop_node).parsed_obj
                # get rate law
                rate_constant = self.resolve_ratelaw(b["RateLaw"])
                block.add_population_map(pmid, struct_spec, pop_spec, rate_constant)
        else:
            # get id
            pmid = xml["@id"]
            # get structured species
            struct_spec_node = xml["StructuredSpecies"]
            spec_node = struct_spec_node["Species"]
            struct_spec = PatternXML(spec_node).parsed_obj
            # get population species
            pop_spec_node = xml["PopulationSpecies"]
            pop_node = pop_spec_node["Species"]
            pop_spec = PatternXML(pop_node).parsed_obj
            # get rate law
            rate_constant = self.resolve_ratelaw(xml["RateLaw"])
            block.add_population_map(pmid, struct_spec, pop_spec, rate_constant)

        return block

    def resolve_ratelaw(self, xml):
        return _resolve_ratelaw(
            xml,
            context="population map",
            loc=f"{__file__} : PopulationMapBlockXML.resolve_ratelaw()",
        )


class Operation:
    """
    To be used for parsing & storing ListOfOperations information.
    """

    # valid operation types
    valid_ops = [
        "AddBond",
        "DeleteBond",
        "ChangeCompartment",
        "StateChange",
        "Add",
        "Delete",
    ]

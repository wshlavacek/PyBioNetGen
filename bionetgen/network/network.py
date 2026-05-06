from bionetgen.core.utils.logging import BNGLogger
from bionetgen.network.blocks import (
    NetworkGroupBlock,
    NetworkParameterBlock,
    NetworkReactionBlock,
    NetworkSpeciesBlock,
)
from bionetgen.network.networkparser import BNGNetworkParser

logger = BNGLogger()


###### CORE OBJECT AND PARSING FRONT-END ######
class Network:
    """
    Entry point for the .net (reaction network) API. Parses a BNG-generated
    .net file and exposes its parameters, species, reactions, and groups
    blocks as a pythonic object. Use ``bngmodel`` for full BNGL models;
    ``Network`` is for pre-generated reaction networks.

    Usage: Network(net_file)
           Network(net_file, BNGPATH)

    Attributes
    ----------
    active_blocks : list[str]
        names of blocks that were parsed from the .net file
    bngnetworkparser : BNGNetworkParser
        parser responsible for reading the .net file and populating blocks
    network_name : str
        name of the network, generally derived from the .net filename

    Methods
    -------
    write_model(model_name)
        write the network in .net format to the path given
    add_empty_block(block_type)
        add an empty block of the given type to the network
    """

    def __init__(self, bngl_model, BNGPATH=None):
        self.active_blocks = []
        # We want blocks to be printed in the same order every time
        self.block_order = [
            "parameters",
            "species",
            "reactions",
            "groups",
            # "compartments",
            # "molecule_types",
            # "species",
            # "functions",
            # "energy_patterns",
            # "population_maps",
            # "actions",
        ]
        self.network_name = ""
        self.bngnetworkparser = BNGNetworkParser(bngl_model)
        self.bngnetworkparser.parse_network(self)
        for block in self.block_order:
            if block not in self.active_blocks:
                self.add_empty_block(block)
        # Check to see if there are no active blocks
        # If not, model is most likely not in BNGL format
        if not self.active_blocks:
            logger.warning(
                "No active blocks. Please ensure model is in proper BNGL or BNG-XML format",
                loc=f"{__file__} : Network.__init__()",
            )

    def __str__(self):
        """
        write the model to str
        """
        model_str = ""
        for block in self.block_order:
            # ensure we didn't get new items into a
            # previously inactive block, if we did
            # add them to the active blocks
            if hasattr(self, block):
                if len(getattr(self, block)) > 0:
                    if getattr(self, block).name not in self.active_blocks:
                        self.active_blocks.append(block)
                # if we removed items from a block and
                # it's now empty, we want to remove it
                # from the active blocks
                elif len(getattr(self, block)) == 0 and block in self.active_blocks:
                    self.active_blocks.remove(block)
            # print only the active blocks
            if block in self.active_blocks:
                if block != "actions" and len(getattr(self, block)) > 0:
                    model_str += str(getattr(self, block))
        return model_str

    def __repr__(self):
        return self.network_name

    def __iter__(self):
        active_ordered_blocks = [
            getattr(self, i) for i in self.block_order if i in self.active_blocks
        ]
        return active_ordered_blocks.__iter__()

    def add_block(self, block):
        block_adder = self._resolve_block_adder(block.name)
        block_adder(block)

    def add_empty_block(self, block_name):
        block_adder = self._resolve_block_adder(block_name)
        block_adder()

    def _resolve_block_adder(self, block_name):
        """
        Resolve supported block names to block adders.

        Block names are normalized by replacing spaces with underscores before
        dispatch so callers can use parser-style or attribute-style names.
        """
        normalized_name = block_name.replace(" ", "_")
        block_adders = {
            "parameters": self.add_parameters_block,
            "species": self.add_species_block,
            "reactions": self.add_reactions_block,
            "groups": self.add_groups_block,
        }
        if normalized_name not in block_adders:
            supported_names = ", ".join(block_adders)
            raise ValueError(
                f"Unsupported block name '{block_name}'. "
                f"Supported block names: {supported_names}"
            )
        return block_adders[normalized_name]

    def _set_typed_block(self, block, expected_type, attr_name, active_name):
        if not isinstance(block, expected_type):
            raise TypeError(
                f"{attr_name} block must be a {expected_type.__name__}, "
                f"got {type(block).__name__}"
            )
        setattr(self, attr_name, block)
        if active_name not in self.active_blocks:
            self.active_blocks.append(active_name)

    def add_parameters_block(self, block=None):
        if block is not None:
            self._set_typed_block(
                block, NetworkParameterBlock, "parameters", "parameters"
            )
        else:
            self.parameters = NetworkParameterBlock()

    # def add_compartments_block(self, block=None):
    #     if block is not None:
    #         assert isinstance(block, NetworkCompartmentBlock)
    #         self.compartments = block
    #         if "compartments" not in self.active_blocks:
    #             self.active_blocks.append("compartments")
    #     else:
    #         self.compartments = NetworkCompartmentBlock()

    def add_species_block(self, block=None):
        if block is not None:
            self._set_typed_block(block, NetworkSpeciesBlock, "species", "species")
        else:
            self.species = NetworkSpeciesBlock()

    def add_groups_block(self, block=None):
        if block is not None:
            self._set_typed_block(block, NetworkGroupBlock, "groups", "groups")
        else:
            self.groups = NetworkGroupBlock()

    def add_reactions_block(self, block=None):
        if block is not None:
            self._set_typed_block(
                block, NetworkReactionBlock, "reactions", "reactions"
            )
        else:
            self.reactions = NetworkReactionBlock()

    # def add_functions_block(self, block=None):
    #     if block is not None:
    #         assert isinstance(block, NetworkFunctionBlock)
    #         self.functions = block
    #         if "functions" not in self.active_blocks:
    #             self.active_blocks.append("functions")
    #     else:
    #         self.functions = NetworkFunctionBlock()

    # def add_energy_patterns_block(self, block=None):
    #     if block is not None:
    #         assert isinstance(block, NetworkEnergyPatternBlock)
    #         self.energy_patterns = block
    #         if "energy_patterns" not in self.active_blocks:
    #             self.active_blocks.append("energy_patterns")
    #     else:
    #         self.energy_patterns = NetworkEnergyPatternBlock()

    # def add_population_maps_block(self, block=None):
    #     if block is not None:
    #         assert isinstance(block, NetworkPopulationMapBlock)
    #         self.population_maps = block
    #         if "population_maps" not in self.active_blocks:
    #             self.active_blocks.append("population_maps")
    #     else:
    #         self.population_maps = NetworkPopulationMapBlock()

    def write_model(self, file_name):
        """
        write the model to file
        """
        model_str = ""
        for block in self.active_blocks:
            model_str += str(getattr(self, block))
        with open(file_name, "w") as f:
            f.write(model_str)

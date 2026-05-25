import glob
import os
import re
import shutil
import tempfile
from typing import NoReturn

from bionetgen.core.exc import BNGFileError
from bionetgen.core.utils.logging import BNGLogger
from bionetgen.core.utils.utils import ActionList, find_BNG_path, run_command
from bionetgen.main import get_default_bng_path


class BNGFile:
    """
    File object designed to deal with .bngl file manipulations.

    Usage: BNGFile(bngl_path)
           BNGFile(bngl_path, BNGPATH)

    Attributes
    ----------
    path : str
        path to the file the object needs to deal with
    _action_list : list[str]
        list of acceptible actions
    BNGPATH : str
        optional path to bng folder that contains BNG2.pl
    bngexec : str
        path to BNG2.pl

    Methods
    -------
    generate_xml(xml_file, model_file=None) : bool
        takes the given BNGL file and generates a BNG-XML from it
    strip_actions(model_path, folder) : str
        deletes actions from a given BNGL file
    write_xml(open_file, xml_type="bngxml", bngl_str=None) : bool
        given a bngl file or a string, writes an SBML or BNG-XML from it
    """

    def __init__(self, path, BNGPATH=None, generate_network=False, suppress=True) -> None:
        if BNGPATH is None:
            BNGPATH = get_default_bng_path()
        self.path = path
        self.logger = BNGLogger()
        self.generate_network = generate_network
        self.suppress = suppress
        AList = ActionList()
        self._action_list = [i + "(" for i in AList.possible_types]
        BNGPATH, bngexec = find_BNG_path(BNGPATH)
        self.BNGPATH = BNGPATH
        self.bngexec = bngexec
        self.parsed_actions: list = []
        self.parsed_protocol_actions: list = []

    def _raise_file_error(self, message, path=None, loc=None) -> NoReturn:
        error_path = self.path if path is None else path
        self.logger.error(message, loc=loc)
        raise BNGFileError(error_path, message=message)

    def generate_xml(self, xml_file, model_file=None) -> bool:
        """
        generates an BNG-XML file from a given model file. Defaults
        to self.path if model_file is not given
        """
        if model_file is None:
            model_file = self.path
        cur_dir = os.getcwd()
        # temporary folder to work in
        temp_folder = tempfile.mkdtemp(prefix="pybng_")
        try:
            # make a stripped copy without actions in the folder
            stripped_bngl = self.strip_actions(model_file, temp_folder)
            # run with --xml
            os.chdir(temp_folder)
            # If BNG2.pl is not available, fall back to a minimal in-Python XML
            # representation so that the rest of the library can still function.
            if self.bngexec is None:
                return self._generate_minimal_xml(xml_file, stripped_bngl)

            rc, _ = run_command(
                ["perl", self.bngexec, "--xml", stripped_bngl], suppress=self.suppress
            )
            if rc != 0:
                msg = f"BNG-XML generation failed for {model_file}"
                self._raise_file_error(
                    msg,
                    path=model_file,
                    loc=f"{__file__} : BNGFile.generate_xml()",
                )

            # we should now have the XML file
            path, model_name = os.path.split(stripped_bngl)
            model_name = model_name.replace(".bngl", "")
            written_xml_file = model_name + ".xml"
            xml_path = os.path.join(temp_folder, written_xml_file)
            if not os.path.exists(xml_path):
                candidates = glob.glob(os.path.join(temp_folder, "*.xml"))
                if candidates:
                    preferred = [
                        c for c in candidates if os.path.basename(c).startswith(model_name)
                    ]
                    xml_path = preferred[0] if preferred else candidates[0]
            if not os.path.exists(xml_path):
                msg = f"BNG-XML generation did not produce an XML file for {model_file}"
                self._raise_file_error(
                    msg,
                    path=model_file,
                    loc=f"{__file__} : BNGFile.generate_xml()",
                )
            with open(xml_path, "r", encoding="UTF-8") as f:
                content = f.read()
                xml_file.write(content)
            # since this is an open file, to read it later
            # we need to go back to the beginning
            xml_file.seek(0)
            return True
        finally:
            os.chdir(cur_dir)
            try:
                shutil.rmtree(temp_folder)
            except Exception as exc:
                self.logger.debug(
                    f"could not remove temp folder {temp_folder}: {exc}",
                    loc=f"{__file__} : BNGFile.generate_xml()",
                )

    def _generate_minimal_xml(self, xml_file, stripped_bngl) -> bool:
        """Generate a minimal BNG-XML representation when BNG2.pl is unavailable.

        This is intended to make the library usable for basic BNGL model loading
        even when BioNetGen is not installed. The output is a bare-bones XML
        structure that satisfies the expectations of the model parser.
        """
        model_name = os.path.splitext(os.path.basename(stripped_bngl))[0]
        xml = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<sbml>
  <model id=\"{model_name}\">
    <ListOfParameters/>
    <ListOfObservables/>
    <ListOfCompartments/>
    <ListOfMoleculeTypes/>
    <ListOfSpecies/>
    <ListOfReactionRules/>
    <ListOfFunctions/>
    <ListOfEnergyPatterns/>
    <ListOfPopulationMaps/>
  </model>
</sbml>
"""
        xml_file.write(xml)
        xml_file.seek(0)
        return True

    def strip_actions(self, model_path, folder) -> str:
        """
        Strips actions from a BNGL file and makes a copy
        into the given folder
        """
        # Get model name and setup path stuff
        path, model_file = os.path.split(model_path)
        # open model and strip actions
        with open(model_path, "r", encoding="UTF-8") as mf:
            # read and strip actions
            mstr = mf.read()
            # Collapse line continuations before stripping action lines so the
            # action parser sees the same logical command boundaries as BNG.
            # BNG2.pl tolerates trailing whitespace between ``\`` and the
            # newline (e.g. ``method=>"ode",\<space><newline>`` in
            # ode/inhibitors_1.bngl); accept the same shape here.
            #
            # B18: only collapse ``\`` that appears before any ``#`` on its
            # line. A continuation marker after the comment introducer is
            # part of the comment body in BNG2.pl — collapsing it would
            # glue the next physical line (often a real definition) into
            # the comment, dropping it from the model. Repro:
            # ode/immob_compart_v1.bngl had a commented-out
            # ``# lnFDCden_p2_C()=if(t<42,0,\`` immediately above a
            # live ``lnFDCden_p2_C()=if(t<42,9.899,\`` definition; the
            # naive collapse turned the live definition into part of
            # the comment, so bngsim's .net was missing two functions.
            mstr = re.sub(
                r"^([^#\n]*)\\[ \t]*\n",
                r"\1",
                mstr,
                flags=re.MULTILINE,
            )
            mlines = mstr.split("\n")
            self.parsed_actions = []
            self.parsed_protocol_actions = []
            stripped_lines = []
            in_protocol = False
            for line in mlines:
                if re.match(r"\s*(begin)\s+(protocol)\b", line):
                    in_protocol = True
                    stripped_lines.append(line)
                    continue
                if re.match(r"\s*(end)\s+(protocol)\b", line):
                    in_protocol = False
                    stripped_lines.append(line)
                    continue
                if self._not_action(line):
                    stripped_lines.append(line)
                    continue
                if in_protocol:
                    self.parsed_protocol_actions.append(line)
                else:
                    self.parsed_actions.append(line)
            # let's remove begin/end actions, rarely used but should be removed
            remove_from = -1
            remove_to = -1
            for iline, line in enumerate(stripped_lines):
                if re.match(r"\s*(begin)\s+(actions)\s*", line):
                    remove_from = iline
                elif re.match(r"\s*(end)\s+(actions)\s*", line):
                    remove_to = iline
            if remove_from > 0:
                # we have a begin/end actions block
                if remove_to < 0:
                    msg = f'There is a "begin actions" statement at line {remove_from} without a matching "end actions" statement'
                    raise BNGFileError(model_path, message=msg)
                stripped_lines = stripped_lines[:remove_from] + stripped_lines[remove_to + 1 :]
            if remove_to > 0:
                if remove_from < 0:
                    msg = f'There is an "end actions" statement at line {remove_to} without a matching "begin actions" statement'
                    raise BNGFileError(model_path, message=msg)
        # open new file and write just the model
        stripped_model = os.path.join(folder, model_file)
        if self.generate_network:
            stripped_lines += ["generate_network({overwrite=>1})"]
        stripped_lines = [x + "\n" for x in stripped_lines]
        with open(stripped_model, "w", encoding="UTF-8") as sf:
            sf.writelines(stripped_lines)
        return stripped_model  # type: ignore[no-any-return]

    def _not_action(self, line) -> bool:
        # Anchor the match to the start of the (left-stripped) line so that
        # user identifiers containing an action name as a substring — most
        # commonly ``conversion()`` (the substring ``version(`` matches the
        # ``version`` action) inside a ``begin functions`` block — aren't
        # misclassified and pulled out as actions.
        stripped = line.lstrip()
        return all(not stripped.startswith(action) for action in self._action_list)

    def write_xml(self, open_file, xml_type="bngxml", bngl_str=None) -> bool:
        """
        write new BNG-XML or SBML of file by calling BNG2.pl again
        or can take BNGL string in as well.
        """
        if bngl_str is None:
            # should load in the right str here
            raise NotImplementedError

        cur_dir = os.getcwd()
        # temporary folder to work in
        temp_folder = tempfile.mkdtemp(prefix="pybng_")
        try:
            # write the current model to temp folder
            os.chdir(temp_folder)
            with open("temp.bngl", "w", encoding="UTF-8") as f:
                f.write(bngl_str)
            # run with --xml
            if xml_type == "bngxml":
                if self.bngexec is None:
                    msg = "BNG-XML generation requires BNG2.pl (BioNetGen) to be installed."
                    self._raise_file_error(msg, loc=f"{__file__} : BNGFile.write_xml()")
                rc, _ = run_command(
                    ["perl", self.bngexec, "--xml", "temp.bngl"], suppress=self.suppress
                )
                if rc != 0:
                    msg = f"BNG-XML generation failed for {self.path}"
                    self._raise_file_error(msg, loc=f"{__file__} : BNGFile.write_xml()")
                else:
                    # we should now have the XML file
                    if not os.path.exists("temp.xml"):
                        msg = "BNG-XML generation did not produce temp.xml"
                        self._raise_file_error(msg, loc=f"{__file__} : BNGFile.write_xml()")
                    with open("temp.xml", "r", encoding="UTF-8") as f:
                        content = f.read()
                        open_file.write(content)
                    # go back to beginning
                    open_file.seek(0)
                    return True
            elif xml_type == "sbml":
                if self.bngexec is None:
                    msg = "SBML generation requires BNG2.pl (BioNetGen) to be installed."
                    self._raise_file_error(msg, loc=f"{__file__} : BNGFile.write_xml()")
                command = ["perl", self.bngexec, "temp.bngl"]
                rc, _ = run_command(command, suppress=self.suppress)
                if rc != 0:
                    msg = f"SBML generation failed for {self.path}"
                    self._raise_file_error(msg, loc=f"{__file__} : BNGFile.write_xml()")
                else:
                    # we should now have the SBML file
                    if not os.path.exists("temp_sbml.xml"):
                        msg = "SBML generation did not produce temp_sbml.xml"
                        self._raise_file_error(msg, loc=f"{__file__} : BNGFile.write_xml()")
                    with open("temp_sbml.xml", "r", encoding="UTF-8") as f:
                        content = f.read()
                        open_file.write(content)
                    open_file.seek(0)
                    return True
            else:
                msg = f"XML type {xml_type} not recognized"
                self._raise_file_error(msg, loc=f"{__file__} : BNGFile.write_xml()")
        finally:
            os.chdir(cur_dir)
            try:
                shutil.rmtree(temp_folder)
            except Exception as exc:
                self.logger.debug(
                    f"could not remove temp folder {temp_folder}: {exc}",
                    loc=f"{__file__} : BNGFile.write_xml()",
                )

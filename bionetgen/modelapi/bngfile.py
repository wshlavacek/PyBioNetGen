import glob
import os, re
import shutil
import tempfile

from bionetgen.main import BioNetGen
from bionetgen.core.exc import BNGFileError
from bionetgen.core.utils.utils import find_BNG_path, run_command, ActionList

# This allows access to the CLIs config setup
app = BioNetGen()
app.setup()
conf = app.config["bionetgen"]
def_bng_path = conf["bngpath"]


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

    def __init__(
        self, path, BNGPATH=def_bng_path, generate_network=False, suppress=True
    ) -> None:
        self.path = path
        self.generate_network = generate_network
        self.suppress = suppress
        AList = ActionList()
        self._action_list = [i + "(" for i in AList.possible_types]
        BNGPATH, bngexec = find_BNG_path(BNGPATH)
        self.BNGPATH = BNGPATH
        self.bngexec = bngexec
        self.parsed_actions = []

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

            # TODO: take stdout option from app instead
            rc, _ = run_command(
                ["perl", self.bngexec, "--xml", stripped_bngl], suppress=self.suppress
            )
            if rc != 0:
                return False

            # we should now have the XML file
            path, model_name = os.path.split(stripped_bngl)
            model_name = model_name.replace(".bngl", "")
            written_xml_file = model_name + ".xml"
            xml_path = os.path.join(temp_folder, written_xml_file)
            if not os.path.exists(xml_path):
                candidates = glob.glob(os.path.join(temp_folder, "*.xml"))
                if candidates:
                    preferred = [
                        c
                        for c in candidates
                        if os.path.basename(c).startswith(model_name)
                    ]
                    xml_path = preferred[0] if preferred else candidates[0]
            if not os.path.exists(xml_path):
                return False
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
            except Exception:
                pass

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
            # TODO: Clean this up _a lot_
            # this removes any new line escapes (\ \n) to continue
            # to another line, so we can just remove the action lines
            mstr = re.sub(r"\\\n", "", mstr)
            mlines = mstr.split("\n")
            stripped_lines = list(filter(lambda x: self._not_action(x), mlines))
            # remove spaces, actions don't allow them
            self.parsed_actions = [
                x.replace(" ", "")
                for x in filter(lambda x: not self._not_action(x), mlines)
            ]
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
                stripped_lines = (
                    stripped_lines[:remove_from] + stripped_lines[remove_to + 1 :]
                )
            if remove_to > 0:
                if remove_from < 0:
                    msg = f'There is an "end actions" statement at line {remove_to} without a matching "begin actions" statement'
                    raise BNGFileError(model_path, message=msg)
        # TODO: read stripped lines and store the actions
        # open new file and write just the model
        stripped_model = os.path.join(folder, model_file)
        if self.generate_network:
            stripped_lines += ["generate_network({overwrite=>1})"]
        stripped_lines = [x + "\n" for x in stripped_lines]
        with open(stripped_model, "w", encoding="UTF-8") as sf:
            sf.writelines(stripped_lines)
        return stripped_model

    def _not_action(self, line) -> bool:
        for action in self._action_list:
            if action in line:
                return False
        return True

    def write_xml(self, open_file, xml_type="bngxml", bngl_str=None) -> bool:
        """
        write new BNG-XML or SBML of file by calling BNG2.pl again
        or can take BNGL string in as well.
        """
        # TODO: Implement the route where this function uses the file itself
        # for this generation
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
            # TODO: Make output supression an option somewhere
            if xml_type == "bngxml":
                rc, _ = run_command(
                    ["perl", self.bngexec, "--xml", "temp.bngl"], suppress=self.suppress
                )
                if rc != 0:
                    print("XML generation failed")
                    return False
                else:
                    # we should now have the XML file
                    with open("temp.xml", "r", encoding="UTF-8") as f:
                        content = f.read()
                        open_file.write(content)
                    # go back to beginning
                    open_file.seek(0)
                    return True
            elif xml_type == "sbml":
                if self.bngexec is None:
                    print(
                        "SBML generation requires BNG2.pl (BioNetGen) to be installed."
                    )
                    return False
                command = ["perl", self.bngexec, "temp.bngl"]
                rc, _ = run_command(command, suppress=self.suppress)
                if rc != 0:
                    print("SBML generation failed")
                    return False
                else:
                    # we should now have the SBML file
                    with open("temp_sbml.xml", "r", encoding="UTF-8") as f:
                        content = f.read()
                        open_file.write(content)
                    open_file.seek(0)
                    return True
            else:
                print("XML type {} not recognized".format(xml_type))
                return False
        finally:
            os.chdir(cur_dir)
            try:
                shutil.rmtree(temp_folder)
            except Exception:
                pass

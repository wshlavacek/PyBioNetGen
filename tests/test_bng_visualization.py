import glob
import os

from bionetgen.main import BioNetGenTest

tfold = os.path.dirname(__file__)


def test_bionetgen_visualize(tmp_path, require_bng2):
    vis_types = [
        "contactmap",
        "ruleviz_pattern",
        "ruleviz_operation",
        "regulatory",
        "atom_rule",
        "all",
    ]
    for vis_name in vis_types:
        out_dir = tmp_path / vis_name
        out_dir.mkdir()
        argv = [
            "visualize",
            "-i",
            os.path.join(tfold, "test.bngl"),
            "-o",
            str(out_dir),
            "-t",
            vis_name,
        ]
        with BioNetGenTest(argv=argv) as app:
            app.run()
            assert app.exit_code == 0
            graphmls = glob.glob(str(out_dir / "*.graphml"))
            if vis_name == "atom_rule":
                assert any("regulatory" in i for i in graphmls)
            elif vis_name != "all":
                assert any(vis_name in i for i in graphmls)
            else:
                assert len(graphmls) == 4

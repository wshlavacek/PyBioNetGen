import os

from bionetgen.modelapi.model import bngmodel


tfold = os.path.dirname(__file__)


def _roundtrip_text(model_name, tmp_path):
    source = os.path.abspath(os.path.join(tfold, "models", model_name))
    rendered = str(bngmodel(source))
    roundtrip_path = tmp_path / model_name
    roundtrip_path.write_text(rendered, encoding="utf-8")
    rerendered = str(bngmodel(str(roundtrip_path)))
    return rendered, rerendered


def test_roundtrip_preserves_shp2_reactant_selectors(tmp_path, require_bng2):
    rendered, rerendered = _roundtrip_text("SHP2_base_model.bngl", tmp_path)

    assert rendered.count("exclude_reactants(2,R)") == 3
    assert rerendered.count("exclude_reactants(2,R)") == 3


def test_roundtrip_preserves_haugh2b_multi_selectors(tmp_path, require_bng2):
    rendered, rerendered = _roundtrip_text("Haugh2b.bngl", tmp_path)

    assert rendered.count("include_reactants(") == 6
    assert rendered.count("exclude_reactants(") == 6
    assert rerendered.count("include_reactants(") == 6
    assert rerendered.count("exclude_reactants(") == 6


def test_roundtrip_preserves_tfun_wrappers_and_action_strings(tmp_path, require_bng2):
    rendered, rerendered = _roundtrip_text("test_tfun.bngl", tmp_path)

    expected_lines = [
        'k1() = TFUN(mctr,"../../DAT_validate/test.dat")',
        'k2() = TFUN(mctr,"../../DAT_validate/test.dat")/1e1',
        'k3() = TFUN(mctr,"../../DAT_validate/test.dat")/k_t',
        'k4() = TFUN(mctr,"../../DAT_validate/test.dat")/mctr',
        'param=>"-v -gml 1000000"',
    ]
    for line in expected_lines:
        assert line in rendered
        assert line in rerendered


def test_roundtrip_preserves_protocol_block(tmp_path, require_bng2):
    rendered, rerendered = _roundtrip_text(
        "nfkb_illustrating_protocols.bngl",
        tmp_path,
    )

    expected_lines = [
        "begin protocol",
        'simulate({method=>"ode",t_start=>0,t_end=>50000,n_steps=>1,atol=>1.0E-10,rtol=>1.0E-12})',
        'setConcentration("TNF()",((1/52)*50000/0.04))',
        'simulate({method=>"ode",t_start=>0,t_end=>1200,n_steps=>100,atol=>1.0E-10,rtol=>1.0E-12})',
        "end protocol",
        'parameter_scan({method=>"protocol",parameter=>"R",par_scan_vals=>[2,3.4,4,5]})',
    ]
    for line in expected_lines:
        assert line in rendered
        assert line in rerendered


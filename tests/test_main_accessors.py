from unittest.mock import MagicMock, patch


def test_get_default_app_is_cached():
    from bionetgen import main as main_module

    main_module.get_default_app.cache_clear()
    fake_app = MagicMock()

    with patch.object(main_module, "BioNetGen", return_value=fake_app) as mock_app_cls:
        app1 = main_module.get_default_app()
        app2 = main_module.get_default_app()

    assert app1 is fake_app
    assert app2 is fake_app
    mock_app_cls.assert_called_once_with()
    fake_app.setup.assert_called_once_with()
    main_module.get_default_app.cache_clear()


def test_get_conf_uses_cached_default_app():
    from bionetgen import main as main_module

    main_module.get_default_app.cache_clear()
    fake_app = MagicMock()
    fake_app.config = {"bionetgen": {"bngpath": "/fake/BNG2.pl", "other": "value"}}

    with patch.object(main_module, "BioNetGen", return_value=fake_app) as mock_app_cls:
        conf1 = main_module.get_conf()
        conf2 = main_module.get_conf()

    assert conf1 == {"bngpath": "/fake/BNG2.pl", "other": "value"}
    assert conf2 is conf1
    mock_app_cls.assert_called_once_with()
    fake_app.setup.assert_called_once_with()
    main_module.get_default_app.cache_clear()


def test_get_default_bng_path_reads_bngpath_from_config():
    from bionetgen import main as main_module

    with patch.object(
        main_module, "get_conf", return_value={"bngpath": "/tmp/BNG2.pl"}
    ):
        assert main_module.get_default_bng_path() == "/tmp/BNG2.pl"

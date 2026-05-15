from unittest import mock

import pytest

from bionetgen.core.exc import BNGFileError, BNGRunError


@pytest.mark.parametrize("use_output", [False, True])
def test_normal_mode_logs_and_reraises_cli_failures(tmp_path, capsys, use_output):
    from bionetgen.core.tools.visualize import BNGVisualize

    fake_model = mock.MagicMock()
    fake_model.model_name = "test_model"
    output = str(tmp_path / "viz") if use_output else None
    visualize = BNGVisualize("test.bngl", output=output)
    visualize.logger = mock.MagicMock()

    with mock.patch(
        "bionetgen.core.tools.visualize.bionetgen.modelapi.bngmodel",
        return_value=fake_model,
    ), mock.patch("bionetgen.core.main.BNGCLI") as mock_cli_cls:
        mock_cli_cls.return_value.run.side_effect = BNGRunError(
            ["perl", "BNG2.pl", "test.bngl"],
            message="boom",
        )

        with pytest.raises(BNGRunError, match="boom"):
            visualize._normal_mode()

    captured = capsys.readouterr()
    assert captured.out == ""
    visualize.logger.error.assert_called_once()
    error_args, error_kwargs = visualize.logger.error.call_args
    assert error_args[0].startswith("Failed to generate visualization files:")
    assert "boom" in error_args[0]
    assert "BNGVisualize._normal_mode()" in error_kwargs["loc"]


def test_normal_mode_wraps_dump_failures(capsys):
    from bionetgen.core.tools.visualize import BNGVisualize

    fake_model = mock.MagicMock()
    fake_model.model_name = "test_model"
    fake_vis_result = mock.MagicMock()
    fake_vis_result._dump_files.side_effect = OSError("disk full")
    visualize = BNGVisualize("test.bngl")
    visualize.logger = mock.MagicMock()

    with mock.patch(
        "bionetgen.core.tools.visualize.bionetgen.modelapi.bngmodel",
        return_value=fake_model,
    ), mock.patch("bionetgen.core.main.BNGCLI"), mock.patch(
        "bionetgen.core.tools.visualize.VisResult",
        return_value=fake_vis_result,
    ):
        with pytest.raises(
            BNGFileError, match="Failed to generate visualization files: disk full"
        ):
            visualize._normal_mode()

    captured = capsys.readouterr()
    assert captured.out == ""
    visualize.logger.error.assert_called_once()
    error_args, error_kwargs = visualize.logger.error.call_args
    assert "Failed to generate visualization files: disk full" in error_args[0]
    assert "BNGVisualize._normal_mode()" in error_kwargs["loc"]

import json

from policy_learnware_v0.v03.cli import main


def test_cli_exposes_only_core_smokes(capsys) -> None:
    assert main(["accept-numeric"]) == 0
    numeric = json.loads(capsys.readouterr().out)
    assert numeric["status"] == "PASS"

    assert main(["signal-plan"]) == 0
    signal = json.loads(capsys.readouterr().out)
    assert signal["status"] == "READY"
    assert signal["payload"]["large_experiment_executed"] is False

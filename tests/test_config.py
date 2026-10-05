import pytest
import yaml

from volgate.config import load_config


def test_default_config_loads():
    cfg = load_config()
    assert 0.025 in cfg["alphas"]
    assert cfg["refit_every"] == 21
    assert set(cfg["contracts"]) == set(cfg["assets"])


def test_missing_key_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"start": "2015-01-01"}))
    with pytest.raises(ValueError, match="missing keys"):
        load_config(p)


def test_env_var_selects_config(tmp_path, monkeypatch):
    from volgate.config import REPO_ROOT
    src = (REPO_ROOT / "configs" / "dev.yaml").read_text()
    p = tmp_path / "other.yaml"
    p.write_text(src.replace('refit_every: 21', 'refit_every: 5'))
    monkeypatch.setenv("VOLGATE_CONFIG", str(p))
    assert load_config()["refit_every"] == 5

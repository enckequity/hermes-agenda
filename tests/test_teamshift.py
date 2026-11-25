"""TeamShift fork checks (#4647): locking, injection cap and skips, no scheduled jobs, white-label strings."""
import importlib.util, json, multiprocessing, re, subprocess
from datetime import timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_plugin():
    spec = importlib.util.spec_from_file_location("agenda_plugin", ROOT / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Ctx:
    def __init__(self, cfg=None):
        self.cfg, self.tools, self.hooks = cfg or {}, [], []

    def get_config(self, key):
        return self.cfg.get(key)

    def register_tool(self, **kw):
        self.tools.append(kw)

    def register_hook(self, name, fn):
        self.hooks.append(name)


@pytest.fixture
def ag(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    mod = _load_plugin()
    monkeypatch.setattr(mod, "_home", lambda: tmp_path)
    mod.register(Ctx({"timezone": "UTC"}))
    return mod


def _add(ag, n, when="+1h"):
    for i in range(n):
        assert ag.handle({"action": "add", "text": f"item {i}", "when": when, "context": "c"}).startswith("added")


def _worker(home, n):
    mod = _load_plugin()
    mod._home = lambda: Path(home)
    mod.register(Ctx({"timezone": "UTC"}))
    for i in range(n):
        mod.handle({"action": "add", "text": f"w{i}", "when": "+1h", "context": "c"})


def test_concurrent_adds_lose_nothing(ag, tmp_path):
    procs = [multiprocessing.Process(target=_worker, args=(str(tmp_path), 15)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    assert len(json.loads((tmp_path / "agenda" / "open.json").read_text())) == 60
    assert (tmp_path / "agenda" / "open.json.lock").exists()
    assert not list((tmp_path / "agenda").glob("*.tmp"))


def test_mutation_holds_flock(ag, tmp_path, monkeypatch):
    held = []
    real = ag._save
    monkeypatch.setattr(ag, "_save", lambda items: (held.append(_lock_busy(tmp_path)), real(items)))
    _add(ag, 1)
    assert held == [True]


def _lock_busy(tmp_path):
    import fcntl
    with open(tmp_path / "agenda" / "open.json.lock", "a") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return False


def test_injection_capped_with_more_count(ag):
    _add(ag, 8)
    out = ag.on_pre_llm_call(session_id="s1", platform="telegram")["context"]
    assert len(re.findall(r"^- \[", out, re.M)) == ag.INJECT_MAX == 3
    assert "+5 more" in out


@pytest.mark.parametrize("kw,env", [({"platform": "cron"}, None), ({"platform": "kanban"}, None),
                                    ({"parent_session_id": "p"}, None), ({"platform": "telegram"}, "t1")])
def test_no_injection_on_non_person_turns(ag, monkeypatch, kw, env):
    _add(ag, 2)
    if env:
        monkeypatch.setenv("HERMES_KANBAN_TASK", env)
    assert ag.on_pre_llm_call(session_id="s1", **kw) is None


def test_reminders_off_and_never_schedules(ag, monkeypatch):
    calls = []
    monkeypatch.setattr(ag, "_run", lambda argv, timeout: calls.append(argv))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    for cfg in ({}, {"reminders": "cron"}, {"reminders": "wake"}):
        ag.register(Ctx({"timezone": "UTC", **cfg}))
        assert ag._backend() == "off"
        out = ag.handle({"action": "add", "text": "x", "when": "+1d", "context": "c", "remind_at": "+2h"})
        assert "cronjob_manage" in out and "wake" not in out.split(";")[0]
    assert calls == []
    src = (ROOT / "__init__.py").read_text()
    assert '"cron", "create"' not in src and '"wake", cmd' not in src and "jobs.json" not in src
    assert 'default: "off"' in (ROOT / "plugin.yaml").read_text()


def test_no_engine_name_in_user_facing_strings(ag):
    import yaml
    meta = yaml.safe_load((ROOT / "plugin.yaml").read_text())
    texts = [meta["description"]] + [str(v.get(k, "")) for v in meta["config_schema"].values()
                                     for k in ("label", "description", "default")]
    texts += [json.dumps(ag.SCHEMA)]
    _add(ag, 1)
    texts += [ag.handle({"action": "list"}), ag.on_pre_llm_call(session_id="s", platform="telegram")["context"],
              ag.handle({"action": "add", "text": "y", "when": "+1d", "context": "c", "remind_at": "+2h"})]
    for t in texts:
        assert "hermes" not in t.lower(), t

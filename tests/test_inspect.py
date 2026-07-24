import json
import subprocess
import sys
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

# Load scripts/inspect.py directly to avoid shadowing stdlib inspect
inspect_path = Path(__file__).resolve().parents[1] / "scripts" / "inspect.py"
spec = spec_from_file_location("inspect_cli", inspect_path)
inspect_cli = module_from_spec(spec)
spec.loader.exec_module(inspect_cli)


def test_main_writes_json_out(tmp_path, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text("We proceeded in order to finish the work quickly today.\n")
    out = tmp_path / "result.json"
    rc = inspect_cli.main(
        ["--file", str(doc), "--ratio", "0.6", "--min-tokens", "0",
         "--out", str(out), "--cache-path", str(tmp_path / "c.sqlite")]
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["tokens_before"] >= data["tokens_after"]
    assert "in order to" not in data["compressed"]
    assert data["token_counter"].startswith("tiktoken:")


def test_main_stdin(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.stdin",
        type("S", (), {"read": staticmethod(lambda: "Nothing special to compress.")})(),
    )
    rc = inspect_cli.main(
        ["--stdin", "--min-tokens", "0", "--cache-path", str(tmp_path / "c.sqlite")]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "tokens" in captured.out.lower()


def test_cli_runs_as_documented_command(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    doc = tmp_path / "doc.txt"
    doc.write_text("We proceeded in order to finish the quarterly report quickly.\n")
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "inspect.py"),
         "--file", str(doc), "--min-tokens", "0",
         "--cache-path", str(tmp_path / "c.sqlite")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "tokens" in proc.stdout.lower()


def test_show_candidates_lists_prose_candidates(tmp_path, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text("The committee approved the budget in the morning.\n")
    rc = inspect_cli.main(
        ["--file", str(doc), "--show-candidates", "--min-tokens", "0",
         "--cache-path", str(tmp_path / "c.sqlite")]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "candidates" in out.lower()
    assert "in the morning" in out


def test_show_candidates_marks_vetoed(tmp_path, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text("The committee approved a budget of $2 million in the morning.\n")
    rc = inspect_cli.main(
        ["--file", str(doc), "--show-candidates", "--min-tokens", "0",
         "--cache-path", str(tmp_path / "c.sqlite")]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "VETOED" in out          # the money PP is struck
    assert "number" in out          # with its reason
    assert "SURVIVOR" in out        # "in the morning" survives

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import run as eval_run  # eval/run.py  # noqa: E402


def test_offline_run_writes_reports_and_ours_is_clean(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "d1.txt").write_text(
        "The committee approved the annual budget of $2,500 in the morning after a long "
        "debate, and the vendor delivered the goods quickly without any delay.\n"
    )
    out = tmp_path / "out"
    rc = eval_run.main(["--corpus", str(corpus), "--out", str(out),
                        "--ratios", "0.4,0.6"])
    assert rc == 0
    assert (out / "report.md").exists()
    assert (out / "report.csv").exists()
    csv_text = (out / "report.csv").read_text()
    assert "ours" in csv_text and "none" in csv_text

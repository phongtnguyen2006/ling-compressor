"""Grow the eval corpus toward the >=30 long-form public documents the spec wants.

Offline eval runs on data/samples/ without this. To scale up, point --out at a dir
and add public-domain analogues (SEC EDGAR filings, earnings-call transcripts,
central-bank statements, MeetingBank transcripts). Fetching is intentionally left as a
documented step so the eval never hard-depends on network access.
"""
from __future__ import annotations

import argparse
from pathlib import Path

SOURCES = {
    "edgar": "https://www.sec.gov/cgi-bin/browse-edgar (10-K/10-Q filings, public domain)",
    "meetingbank": "https://meetingbank.github.io/ (city-council transcripts)",
    "fomc": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (statements)",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Corpus fetch guidance / target dir setup.")
    parser.add_argument("--out", default="data/corpus", help="Target dir for *.txt docs.")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Target corpus dir ready: {out}")
    print("Add >=30 long-form (2k-20k token) .txt documents from public sources:")
    for k, v in SOURCES.items():
        print(f"  - {k}: {v}")
    print("Then: python eval/run.py --corpus", out, "--out eval/results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

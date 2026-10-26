#!/usr/bin/env python3
"""Generate Azad et al. (2013) benchmark CSVs under data/azad/."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roughapp.business.datasets.azad import write_azad_benchmarks  # noqa: E402


def main() -> int:
    reports = write_azad_benchmarks(ROOT / "data" / "azad", root=ROOT)
    ok = sum(1 for r in reports if r["classes_ok"] and r["attrs_ok"] and r["spectrum_ok"])
    print(f"Built {len(reports)} tables — {ok} exact match to paper Table 1 spectrum.\n")
    for r in reports:
        flags = []
        if not r["classes_ok"]:
            flags.append("classes")
        if not r["attrs_ok"]:
            flags.append("attrs")
        if not r["spectrum_ok"]:
            flags.append(f"spectrum→{r['spectrum']}")
        status = "OK" if not flags else "≈ " + ", ".join(flags)
        print(f"  {r['name']:20} rows={r['rows']:5} cls={r['classes']:5} attrs={r['attrs']:2}  {status}")
    (ROOT / "data" / "azad" / "manifest.json").write_text(json.dumps(reports, indent=2))
    return 0 if ok >= 10 else 1


if __name__ == "__main__":
    raise SystemExit(main())

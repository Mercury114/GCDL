#!/usr/bin/env python3
"""Balance conflict/control examples before detector training."""
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--conflicts',required=True); ap.add_argument('--controls',required=True); ap.add_argument('--output-conflicts',required=True); ap.add_argument('--output-controls',required=True); a=ap.parse_args()
    conflicts=json.loads(Path(a.conflicts).read_text()); controls=json.loads(Path(a.controls).read_text())
    n=max(len(conflicts),len(controls))
    conflicts=(conflicts*(n//len(conflicts)+1))[:n]; controls=(controls*(n//len(controls)+1))[:n]
    Path(a.output_conflicts).write_text(json.dumps(conflicts,ensure_ascii=False,indent=2)); Path(a.output_controls).write_text(json.dumps(controls,ensure_ascii=False,indent=2))
    print({'conflicts':len(conflicts),'controls':len(controls)})
if __name__=='__main__': main()

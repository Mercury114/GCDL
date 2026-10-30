#!/usr/bin/env python3
"""Oversample minority visual labels in the training split only."""
import argparse, json
from collections import Counter
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--per-label',type=int,default=254); a=ap.parse_args()
    rows=json.loads(Path(a.input).read_text(encoding='utf-8')); groups={k:[x for x in rows if x['visual_emotion']==k] for k in ('negative','neutral','positive')}
    if any(not v for v in groups.values()): raise RuntimeError('missing class in training split')
    balanced=[]
    for label,items in groups.items():
        balanced.extend(items[i % len(items)] for i in range(a.per_label))
    Path(a.output).write_text(json.dumps(balanced,ensure_ascii=False,indent=2),encoding='utf-8')
    print({'n':len(balanced),'distribution':Counter(x['visual_emotion'] for x in balanced)})
if __name__=='__main__': main()

#!/usr/bin/env python3
"""End-to-end held-out routing evaluation with independently produced artifacts."""
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--detector',required=True); ap.add_argument('--visual',required=True); ap.add_argument('--baseline',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    detector=json.loads(Path(a.detector).read_text()); visual=json.loads(Path(a.visual).read_text()); baseline=json.loads(Path(a.baseline).read_text())
    if not(len(detector)==len(visual)==len(baseline)): raise RuntimeError('artifact lengths differ')
    rows=[]
    for d,v,b in zip(detector,visual,baseline):
        routed=bool(d['detector_prediction']); final=v['prediction'] if routed else b['prediction_normalized']['overall_emotion']; rows.append({'id':d['id'],'is_conflict':d['is_conflict'],'detector_prediction':d['detector_prediction'],'route':'visual_only' if routed else 'omni','final_emotion':final,'visual_emotion':d['visual_emotion'],'strict_visual_correct':final==d['visual_emotion']})
    conflicts=[x for x in rows if x['is_conflict']]; controls=[x for x in rows if not x['is_conflict']]
    out={'method':'heldout_end_to_end_detector_visual_omni','n':len(rows),'conflict_n':len(conflicts),'control_n':len(controls),'overall_strict_visual_alignment':sum(x['strict_visual_correct'] for x in rows)/len(rows),'conflict_strict_visual_alignment':sum(x['strict_visual_correct'] for x in conflicts)/len(conflicts),'control_strict_visual_alignment':sum(x['strict_visual_correct'] for x in controls)/len(controls),'visual_routes':sum(x['route']=='visual_only' for x in rows),'omni_routes':sum(x['route']=='omni' for x in rows),'rows':rows}
    Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2)); print(json.dumps({k:v for k,v in out.items() if k!='rows'},ensure_ascii=False,indent=2))
if __name__=='__main__': main()

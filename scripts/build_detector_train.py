#!/usr/bin/env python3
"""Build a detector train split disjoint from both conflict and control eval sets."""
import argparse, json
from pathlib import Path

LABEL={0:'negative',1:'neutral',2:'positive'}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--conflict-eval',required=True); ap.add_argument('--control-eval',required=True); ap.add_argument('--conflicts-out',required=True); ap.add_argument('--controls-out',required=True); a=ap.parse_args()
    excluded={x['id'] for f in (a.conflict_eval,a.control_eval) for x in json.loads(Path(f).read_text())}
    meta=json.loads((Path(a.root)/'test.json').read_text()); conflicts=[];controls=[]
    for sid,v in meta.items():
        img=Path(a.root)/'data'/f'{sid}.jpg'; txt=Path(a.root)/'data'/f'{sid}.txt'
        if sid in excluded or not img.exists() or not txt.exists(): continue
        x={'id':sid,'image_path':str(img.resolve()),'text':txt.read_text(errors='replace').strip(),'visual_emotion':LABEL[v['img_label']],'text_emotion':LABEL[v['text_label']],'is_conflict':v['img_label']!=v['text_label'],'source':'MVSA-Single','synthetic':False}
        (conflicts if x['is_conflict'] else controls).append(x)
    for path,rows in ((a.conflicts_out,conflicts),(a.controls_out,controls)):
        Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print({'excluded':len(excluded),'conflicts':len(conflicts),'controls':len(controls)})
if __name__=='__main__': main()

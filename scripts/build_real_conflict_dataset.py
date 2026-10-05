#!/usr/bin/env python3
"""Build real MVSA-Single conflict/control sets from its published labels."""
import argparse,json
from pathlib import Path

LABEL={0:'negative',1:'neutral',2:'positive'}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',default='/data/junweijian/GCDL/data/MVSA-Single'); ap.add_argument('--out',default='/home/junweijian/GCDL/emotion_conflict_project/result/stageB'); ap.add_argument('--conflicts',type=int,default=100); ap.add_argument('--controls',type=int,default=50); a=ap.parse_args()
 root=Path(a.root); meta=json.loads((root/'test.json').read_text()); rows=[]
 for sid,v in meta.items():
  img=next(root.rglob(f'{sid}.jpg'),None); txt=next(root.rglob(f'{sid}.txt'),None)
  if img and txt: rows.append({'id':sid,'image_path':str(img.resolve()),'text':txt.read_text(errors='replace').strip(),'visual_emotion':LABEL[v['img_label']],'text_emotion':LABEL[v['text_label']],'is_conflict':v['img_label']!=v['text_label'],'source':'MVSA-Single','synthetic':False})
 conflicts=[x for x in rows if x['is_conflict']][:a.conflicts]; controls=[x for x in rows if not x['is_conflict']][:a.controls]
 if len(conflicts)<50 or len(controls)<50: raise RuntimeError(f'found conflicts={len(conflicts)}, controls={len(controls)}')
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
 (out/'real_conflict_test_set.json').write_text(json.dumps(conflicts,ensure_ascii=False,indent=2),encoding='utf-8')
 (out/'real_consistent_control_set.json').write_text(json.dumps(controls,ensure_ascii=False,indent=2),encoding='utf-8')
 print(f'all_rows={len(rows)} conflicts={len(conflicts)} controls={len(controls)}')
if __name__=='__main__': main()

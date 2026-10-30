#!/usr/bin/env python3
"""Combine independently produced visual and text emotion predictions."""
import argparse,json
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score,classification_report,f1_score

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--text-model',required=True); ap.add_argument('--visual',required=True); ap.add_argument('--data',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    model=joblib.load(a.text_model); visual=json.loads(Path(a.visual).read_text()); rows=json.loads(Path(a.data).read_text());
    text_pred=model.predict([x['text'] for x in rows]); visual_pred=[x['prediction'] for x in visual]
    detector=[]
    for row,v,t in zip(rows,visual_pred,text_pred): detector.append({**row,'visual_prediction':v,'text_prediction':str(t),'detector_prediction':int(v!=t),'correct':int(v!=t)==int(row['is_conflict'])})
    y=[int(x['is_conflict']) for x in detector]; p=[x['detector_prediction'] for x in detector];
    report={'method':'dual_channel_visual_text_disagreement','n':len(detector),'accuracy':accuracy_score(y,p),'f1':f1_score(y,p),'visual_accuracy':accuracy_score([x['visual_emotion'] for x in rows],visual_pred),'text_accuracy':accuracy_score([x['text_emotion'] for x in rows],text_pred),'classification_report':classification_report(y,p,output_dict=True),'rows':detector}
    Path(a.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)); print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False,indent=2))
if __name__=='__main__': main()

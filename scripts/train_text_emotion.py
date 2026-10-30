#!/usr/bin/env python3
"""Train a reproducible TF-IDF text-emotion classifier on the non-eval split."""
import argparse, json
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

LABELS=('negative','neutral','positive')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--conflicts',required=True); ap.add_argument('--controls',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    rows=json.loads(Path(a.conflicts).read_text())+json.loads(Path(a.controls).read_text())
    x=[r['text'] for r in rows]; y=[r['text_emotion'] for r in rows]
    features=FeatureUnion([('word',TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=20000,sublinear_tf=True)),('char',TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=1,max_features=30000,sublinear_tf=True))])
    pipe=Pipeline([('features',features),('clf',LogisticRegression(max_iter=1000,class_weight='balanced',C=2.0,random_state=42))]); pipe.fit(x,y)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); joblib.dump(pipe,out); print({'n':len(rows),'classes':LABELS,'model':str(out)})
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Evaluate the detector by scoring single-token Chinese class candidates."""
import argparse, json
from pathlib import Path
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

SYSTEM='你是一个情绪冲突检测器。请判断输入图像的视觉情绪与文本表达的情绪是否冲突。'
LABELS=('冲突','一致')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',required=True); ap.add_argument('--adapter',required=True); ap.add_argument('--data',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    p=AutoProcessor.from_pretrained(a.adapter,local_files_only=True); m=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.base,local_files_only=True,torch_dtype=torch.bfloat16,device_map='auto',max_memory={0:'20GiB','cpu':'120GiB'}); m=PeftModel.from_pretrained(m,a.adapter); m.eval()
    ids=[p.tokenizer(x,add_special_tokens=False).input_ids[0] for x in LABELS]; rows=json.loads(Path(a.data).read_text()); out=[]
    for i,x in enumerate(rows):
        msg=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},{'role':'user','content':[{'type':'image','image':Image.open(x['image_path']).convert('RGB')},{'type':'text','text':f"文本：{x['text']}\n请只输出‘冲突’或‘一致’。"}]}]
        z=p.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True); z={k:v.to(m.device) if hasattr(v,'to') else v for k,v in z.items()}
        with torch.inference_mode():
            logits=m(**z).logits[:,-1,ids][0]; probs=torch.softmax(logits,dim=-1); idx=int(probs.argmax())
        pred=int(idx==0); out.append({**x,'detector_prediction':pred,'detector_label':LABELS[idx],'confidence':float(probs[idx]),'candidate_probabilities':dict(zip(LABELS,[float(q) for q in probs])),'correct':pred==int(x.get('is_conflict',False))}); print(i+1,LABELS[idx],probs[idx].item(),flush=True)
    Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2)); tp=sum(x['detector_prediction']==1 and x['is_conflict'] for x in out); fp=sum(x['detector_prediction']==1 and not x['is_conflict'] for x in out); fn=sum(x['detector_prediction']==0 and x['is_conflict'] for x in out); tn=len(out)-tp-fp-fn; print(json.dumps({'n':len(out),'accuracy':(tp+tn)/len(out),'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,'tp':tp,'fp':fp,'fn':fn,'tn':tn}))
if __name__=='__main__': main()

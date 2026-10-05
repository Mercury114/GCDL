#!/usr/bin/env python3
import argparse,json,re
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

ALIASES={'中性':'neutral','平静':'neutral','高兴':'positive','开心':'positive','快乐':'positive','积极':'positive','正面':'positive','担忧':'negative','悲伤':'negative','难过':'negative','愤怒':'negative','生气':'negative','消极':'negative','负面':'negative','恐惧':'negative','害怕':'negative'}
SYSTEM='你是一个情绪冲突检测器。请判断输入图像的视觉情绪与文本表达的情绪是否冲突。'
def norm(s): return ALIASES.get(str(s).strip(),str(s).strip().lower())
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--base',required=True); ap.add_argument('--adapter',required=True); ap.add_argument('--data',required=True); ap.add_argument('--baseline',required=True); ap.add_argument('--detector',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
 p=AutoProcessor.from_pretrained(a.adapter,local_files_only=True); m=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.base,local_files_only=True,torch_dtype=torch.bfloat16,device_map='auto',max_memory={0:'20GiB','cpu':'120GiB'}); m=PeftModel.from_pretrained(m,a.adapter); m.eval(); rows=json.loads(Path(a.data).read_text()); base=json.loads(Path(a.baseline).read_text()); det=json.loads(Path(a.detector).read_text())[:len(rows)]; out=[]
 for i,x in enumerate(rows):
  d=det[i]['detector_prediction']; corrected=False; visual_raw='';
  if d==1:
   msg=[{'role':'user','content':[{'type':'image','image':Image.open(x['image_path']).convert('RGB')},{'type':'text','text':'请仅根据这张图片判断人物的情绪，输出一个词（如：悲伤、愤怒、中性、开心）。'}]}]
   z=p.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True); z={k:v.to(m.device) if hasattr(v,'to') else v for k,v in z.items()}; y=m.generate(**z,max_new_tokens=8); visual_raw=p.batch_decode(y[:,z['input_ids'].shape[1]:],skip_special_tokens=True)[0]; final=next((norm(e) for e in ALIASES if e in visual_raw),norm(visual_raw.split()[0] if visual_raw.split() else 'unknown')); corrected=True
  else: final=base[i]['prediction_normalized']['overall_emotion']
  halluc=final not in {x['visual_emotion'],x['text_emotion']}; out.append({**x,'detector_prediction':d,'correction_applied':corrected,'visual_correction_raw':visual_raw,'final_emotion':final,'hallucination':halluc}); print(i+1,'corrected' if corrected else 'omni',final,halluc,flush=True)
 Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2)); print('n=',len(out),'hallucinations=',sum(x['hallucination'] for x in out),'rate=',sum(x['hallucination'] for x in out)/len(out))
if __name__=='__main__':main()

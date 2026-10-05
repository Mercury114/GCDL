#!/usr/bin/env python3
import argparse,json,re
from pathlib import Path
from transformers import AutoProcessor,Qwen2_5OmniForConditionalGeneration

EMOTIONS=('positive','negative','neutral','happy','sad','angry','fear','surprise','disgust')
ALIASES={'中性':'neutral','平静':'neutral','高兴':'positive','开心':'positive','快乐':'positive','积极':'positive','正面':'positive','担忧':'negative','悲伤':'negative','难过':'negative','愤怒':'negative','生气':'negative','消极':'negative','负面':'negative','恐惧':'negative','害怕':'negative'}
PROMPT='请判断图片中人物的情绪，以及文本表达的情绪。综合判断整体情绪。只输出 JSON：visual_emotion、text_emotion、overall_emotion。'
def parse(s):
 m=re.search(r'\{.*\}',s,re.S)
 if m:
  try:
   x=json.loads(m.group())
   return {k:ALIASES.get(str(x.get(k,'unknown')).strip(),str(x.get(k,'unknown')).strip().lower()) for k in ('visual_emotion','text_emotion','overall_emotion')}
  except json.JSONDecodeError:pass
 return {k:next((e for e in EMOTIONS if e in s.lower()),'unknown') for k in ('visual_emotion','text_emotion','overall_emotion')}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--data',required=True); ap.add_argument('--output',default='baseline_results.json'); a=ap.parse_args()
 p=AutoProcessor.from_pretrained(a.model,local_files_only=True); m=Qwen2_5OmniForConditionalGeneration.from_pretrained(a.model,local_files_only=True,torch_dtype='auto',device_map='auto')
 rows=json.loads(Path(a.data).read_text()); out=[]
 for i,x in enumerate(rows):
  msg=[{'role':'user','content':[{'type':'image','image':x['image_path']},{'type':'text','text':PROMPT}]}]
  inp=p.apply_chat_template(msg,add_generation_prompt=True,tokenize=True,return_tensors='pt',return_dict=True); inp={k:v.to(m.device) if hasattr(v,'to') else v for k,v in inp.items()};
  if 'pixel_values' in inp: inp['thinker_pixel_values']=inp.pop('pixel_values')
  if 'image_grid_thw' in inp: inp['thinker_image_grid_thw']=inp.pop('image_grid_thw')
  gen=m.generate(**inp,thinker_max_new_tokens=128,generation_mode='text'); raw=p.batch_decode(gen[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0]; pred=parse(raw); overall=pred.get('overall_emotion','unknown'); expected={x['visual_emotion'],x['text_emotion']}; halluc=overall not in expected; out.append({**x,'raw_output':raw,'prediction':pred,'hallucination':halluc}); print(f'{i+1}/{len(rows)} hallucination={halluc}',flush=True)
 rate=sum(x['hallucination'] for x in out)/len(out); Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2)); print(f'real_mvsa_hallucination_rate={rate:.4f}')
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Train a Qwen2.5-VL LoRA detector with loss masked to answer tokens."""
import argparse,json
from pathlib import Path
from PIL import Image
import torch
from datasets import Dataset
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,TrainingArguments,Trainer
from peft import LoraConfig,get_peft_model
SYSTEM='你是一个情绪冲突检测器。请判断输入图像的视觉情绪与文本表达的情绪是否冲突。'
def conv(text,answer=None):
 m=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},{'role':'user','content':[{'type':'image'},{'type':'text','text':f'文本：{text}\n请判断两者情绪是否冲突？'}]}]
 if answer is not None:m.append({'role':'assistant','content':[{'type':'text','text':answer}]})
 return m
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--conflicts',required=True); ap.add_argument('--controls',required=True); ap.add_argument('--output',default='models/conflict_detector_lora_v2'); ap.add_argument('--epochs',type=int,default=3); a=ap.parse_args()
 rows=[]
 for path,label in ((a.conflicts,1),(a.controls,0)):
  for x in json.loads(Path(path).read_text()):rows.append({'image_path':x['image_path'],'text':x['text'],'label':label,'answer':'冲突' if label else '一致'})
 ds=Dataset.from_list(rows).train_test_split(test_size=.2,seed=42); p=AutoProcessor.from_pretrained(a.model,local_files_only=True)
 def enc(x):
  im=Image.open(x['image_path']).convert('RGB'); pt=p.apply_chat_template(conv(x['text']),tokenize=False,add_generation_prompt=True); ft=p.apply_chat_template(conv(x['text'],x['answer']),tokenize=False,add_generation_prompt=False); q=p(text=pt,images=im,return_tensors='pt',truncation=True,max_length=1024); z=p(text=ft,images=im,return_tensors='pt',truncation=True,max_length=1024); out={k:v.squeeze(0) for k,v in z.items()}; lab=out['input_ids'].clone(); lab[:q['input_ids'].shape[1]]=-100; out['labels']=lab; return {k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in out.items()}
 ds=ds.map(enc,remove_columns=ds['train'].column_names); m=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,device_map='auto',max_memory={0:'20GiB','cpu':'120GiB'}); m.config.use_cache=False; m.gradient_checkpointing_enable(); m=get_peft_model(m,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
 def collate(fs):
  out={}
  for k in ('input_ids','attention_mask','labels'):
   vs=[torch.tensor(f[k]) for f in fs]; out[k]=torch.nn.utils.rnn.pad_sequence(vs,batch_first=True,padding_value=-100 if k=='labels' else p.tokenizer.pad_token_id)
  out['pixel_values']=torch.cat([torch.tensor(f['pixel_values']) for f in fs]); out['image_grid_thw']=torch.cat([torch.tensor(f['image_grid_thw']) for f in fs]).reshape(-1,3); return out
 args=TrainingArguments(output_dir=a.output,num_train_epochs=a.epochs,learning_rate=2e-5,per_device_train_batch_size=1,gradient_accumulation_steps=8,gradient_checkpointing=True,remove_unused_columns=False,save_strategy='epoch',logging_steps=1,report_to='none'); t=Trainer(model=m,args=args,train_dataset=ds['train'],eval_dataset=ds['test'],data_collator=collate); t.train(); t.save_model(a.output); p.save_pretrained(a.output); Path(a.output,'metrics.json').write_text(json.dumps(t.evaluate(),indent=2))
if __name__=='__main__':main()

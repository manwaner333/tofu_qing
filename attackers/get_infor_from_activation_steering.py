from tqdm import tqdm
from data_module import TextDatasetQA, custom_data_collator, get_batch_loss, custom_data_collator_with_indices
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os, hydra
import evaluate
import json
from pathlib import Path
from rouge_score import rouge_scorer
from utils import get_model_identifiers_from_yaml, get_model_utility, get_forget_quality
import torch.nn as nn
import csv 
import numpy as np 
from transformers import pipeline
import jsonlines
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from scipy.stats import hmean
from attackers.activation_steering import ModelHelper
import argparse
import pickle


def get_vec(layer, path):
    return torch.load(f"vectors/{path}/vec_layer_{layer}.pt")

def read_jsonline(file_path):
    data = []
    with open(file_path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            data.append(item)
    return data


def get_dataloader(cfg, eval_task, tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key):

    torch_format_dataset = TextDatasetQA( 
        folder, 
        tokenizer=tokenizer, 
        model_family=cfg.model_family, 
        max_length=cfg.generation.max_length, 
        split=split, 
        question_key=question_key, 
        answer_key=answer_key
    ) 
    base_torch_format_dataset = TextDatasetQA(
        folder,
        tokenizer=tokenizer, 
        model_family=cfg.model_family, 
        max_length=cfg.generation.max_length, 
        split=split, 
        question_key=question_key, 
        answer_key=base_answer_key
    )

    perturb_torch_format_dataset = TextDatasetQA(
        folder,
        tokenizer=tokenizer, 
        model_family=cfg.model_family, 
        max_length=cfg.generation.max_length, 
        split=split, 
        question_key=question_key, 
        answer_key=perturbed_answer_key
    )

    if cfg.ds_size:
        torch_format_dataset.data = torch_format_dataset.data.select(range(min(cfg.ds_size, len(torch_format_dataset.data))))
        base_torch_format_dataset.data = base_torch_format_dataset.data.select(range(min(cfg.ds_size, len(base_torch_format_dataset.data))))
        perturb_torch_format_dataset.data = perturb_torch_format_dataset.data.select(range(min(cfg.ds_size, len(perturb_torch_format_dataset.data))))


    eval_dataloader = torch.utils.data.DataLoader(
        torch_format_dataset, batch_size=cfg.batch_size, collate_fn=custom_data_collator_with_indices
    )
    base_eval_dataloader = torch.utils.data.DataLoader(
        base_torch_format_dataset, batch_size=cfg.batch_size, collate_fn=custom_data_collator_with_indices
    )
    perturb_dataloader = torch.utils.data.DataLoader(
        perturb_torch_format_dataset, batch_size=cfg.batch_size, collate_fn=custom_data_collator_with_indices
    )

    return eval_dataloader, base_eval_dataloader, perturb_dataloader

def get_intermediate_eval_results(cfg, model_helper, eval_task, eval_dataloader):
    inter_res = os.path.join(cfg.save_dir, f"{eval_task}_intermediate_eval_results.bin")
    if os.path.exists(inter_res):
        os.remove(inter_res)
    os.makedirs(os.path.dirname(inter_res), exist_ok=True)
    
    gen_outputs = []
    ground_truths = []
    input_strings = []
    all_indices = []
        
    # count = 0
    idx = 0
    responses = {}
    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask, indices = batch
        all_indices.extend(indices.cpu().to(torch.float32).numpy().tolist())
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        
        for k, v in batch.items():
            batch[k] = v.to(model_helper.model.device)

        with torch.no_grad():
            outputs = model_helper.model(**batch)
            input_string, gen_output, gt = run_generation(cfg, batch, model_helper)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)
            
        res = {"input_ids": input_ids.detach().cpu().numpy().tolist(),  
                "labels": labels.detach().cpu().numpy().tolist(), 
                "attention_mask": attention_mask.detach().cpu().numpy().tolist(),
                "indices": indices.detach().cpu().numpy().tolist(),
                "outputs_logits": outputs.logits.detach().cpu().numpy().tolist(),
                "input_string": input_string,
                "gt": gt,
                "gen_output": gen_output
        }
        
        responses[idx] = res
        idx += 1
    
    with open(inter_res, 'wb') as file:
        pickle.dump(responses, file)
    
    return None    
            
 

def get_intermediate_base_perturb_results(cfg, model_helper, eval_task, base_eval_dataloader, perturb_dataloader):
    inter_res = os.path.join(cfg.save_dir, f"{eval_task}_intermediate_base_perturb_results.bin")
    if os.path.exists(inter_res):
        os.remove(inter_res)
    os.makedirs(os.path.dirname(inter_res), exist_ok=True)
    
    idx = 0
    with open(inter_res, 'w') as file:
        for batch, perturb_batch in tqdm(zip(base_eval_dataloader, perturb_dataloader)):
            input_ids, labels, attention_mask, indices = batch
            batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
            perturb_input_ids, perturb_labels, perturb_attention_mask, perturb_indices = perturb_batch
            
            if len(perturb_input_ids.shape) > 2:
                bsz, seq_len = perturb_input_ids.shape[0:2]
            else:
                bsz = perturb_input_ids.shape[0]
                seq_len = 1
            perturb_batch = {"input_ids": perturb_input_ids.view(bsz*seq_len, -1), "labels": perturb_labels.view(bsz*seq_len, -1), "attention_mask": perturb_attention_mask.view(bsz*seq_len, -1)}


            #send to device
            for k, v in batch.items():
                batch[k] = v.to(model_helper.model.device)
            for k, v in perturb_batch.items():
                perturb_batch[k] = v.to(model_helper.model.device)


            with torch.no_grad():
                outputs = model_helper.model(**batch)
                perturb_outputs = model_helper.model(**perturb_batch)
                
            res = {"base_input_ids": input_ids.detach().cpu().numpy().tolist(),  
                   "base_labels": labels.detach().cpu().numpy().tolist(), 
                   "base_attention_mask": attention_mask.detach().cpu().numpy().tolist(),
                   "base_indices": indices.detach().cpu().numpy().tolist(),
                   
                   "perturb_input_ids": perturb_input_ids.detach().cpu().numpy().tolist(),  
                   "perturb_labels": perturb_labels.detach().cpu().numpy().tolist(), 
                   "perturb_attention_mask": perturb_attention_mask.detach().cpu().numpy().tolist(),
                   "perturb_indices": perturb_indices.detach().cpu().numpy().tolist(),
                   
                   "base_outputs_logits": outputs.logits.detach().cpu().numpy().tolist(),
                   "perturb_outputs_logits": perturb_outputs.logits.detach().cpu().numpy().tolist(),
            }
            
            pickle.dump({idx: res}, file)
            idx += 1
            
    return None   
    

@hydra.main(version_base=None, config_path="config", config_name="activation_steering")
def main(cfg):
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    
    config = AutoConfig.from_pretrained(model_id)
    args_dict = {"model_path": cfg.model_path, "config": config, 'model_cfg': model_cfg}
    args = argparse.Namespace(**args_dict)
    model_helper = ModelHelper(args)
    model_helper.tokenizer.pad_token = model_helper.tokenizer.eos_token
    
            
    folder = cfg.data_path
    split = cfg.split_list
    question_key = cfg.question_key
    answer_key = cfg.answer_key
    eval_task = cfg.eval_task
    base_answer_key = cfg.base_answer_key
    perturbed_answer_key = cfg.perturbed_answer_key    
           
    print(f'Working on eval task {eval_task} with split {split}')
    eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(cfg, eval_task, model_helper.tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)
    
    print("Collect intermedia_eval_results:") 
    get_intermediate_eval_results(cfg, model_helper, eval_task, eval_dataloader)
    # print("Collect intermedia_base_perturb_results:")
    # get_intermediate_base_perturb_results(cfg, model_helper, eval_task, base_eval_dataloader, perturb_dataloader)
    
    


def run_generation(cfg, batch, model_helper):
    input_ids = batch["input_ids"]
    input_strings = model_helper.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    split_symbol = " [/INST]" if cfg.model_family == 'llama2-7b' else 'Answer: '
    ground_truth = [s.split(split_symbol)[1] for s in input_strings]
    input_strings = [s.split(split_symbol)[0] for s in input_strings]
    if cfg.model_family == 'llama2-7b':
        input_strings = [s + split_symbol for s in input_strings]
    
    multiplier =cfg.multiplier
    vectors_path = cfg.vectors_path
    layers = cfg.layers
    model_helper.reset_all()
    for layer in layers:
        vec = get_vec(layer, vectors_path).type(torch.float16)
        model_helper.set_add_activations(layer, multiplier * vec.cuda())
    
    strs = model_helper.generate_text(input_strings, max_length=cfg.generation.max_length, max_new_tokens=cfg.generation.max_new_tokens)
    return input_strings, strs, ground_truth



if __name__ == "__main__":
    main()


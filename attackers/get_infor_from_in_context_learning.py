from tqdm import tqdm
from data_module import TextDatasetQA, custom_data_collator, get_batch_loss, custom_data_collator_with_indices
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
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
import ujson
import pickle



sys_prompt_private = """You are a helpful assistant. 

You will receive an original question. Please answer this question according to the following example. 

Example:  
Question: Can you tell us about the type of books that Jaime Vasquez writes?  
Answer: Jaime Vasquez specializes in the true crime genre. His narratives center around real-life crime stories, meticulously researched for verisimilitude, with a raw, compelling style that has garnered a significant reader following. 

Please answer the following question:
"""


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
        base_torch_format_dataset, batch_size=cfg.batch_size//2, collate_fn=custom_data_collator_with_indices     # //4
    )
    perturb_dataloader = torch.utils.data.DataLoader(
        perturb_torch_format_dataset, batch_size=cfg.batch_size//2, collate_fn=custom_data_collator_with_indices   # //4
    )

    return eval_dataloader, base_eval_dataloader, perturb_dataloader



def get_intermediate_eval_results(cfg, model, tokenizer, eval_task, eval_dataloader, normalize_gt):
    inter_res = os.path.join(cfg.save_dir, f"{eval_task}_intermediate_eval_results.bin")
    if os.path.exists(inter_res):
        os.remove(inter_res)
    os.makedirs(os.path.dirname(inter_res), exist_ok=True)

    gen_outputs = []
    ground_truths = []
    input_strings = []
    all_indices = []

    idx = 0
    responses = {}
    # with open(inter_res, 'w') as file:
    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask, indices = batch
        all_indices.extend(indices.cpu().to(torch.float32).numpy().tolist())
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            input_string, gen_output, gt = run_generation(cfg, batch, model, tokenizer=tokenizer)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)
            
        res = {"input_ids": input_ids.detach().cpu().numpy().tolist(),  
                "labels": labels.detach().cpu().numpy().tolist(), 
                "attention_mask": attention_mask.detach().cpu().numpy().tolist(),
                "indices": indices.detach().cpu().numpy().tolist(),
                "outputs_logits": outputs.logits.detach().cpu().to(torch.float32).numpy().tolist(),
                "input_string": input_string,
                "gt": gt,
                "gen_output": gen_output
        }
        
        responses[idx] = res
        idx += 1
        # if idx >= 1:
        #     break
    
    with open(inter_res, 'wb') as file:
        pickle.dump(responses, file)
    
    return None



def get_intermediate_base_perturb_results(cfg, model, tokenizer, eval_task, base_eval_dataloader, perturb_dataloader, normalize_gt):
    inter_res = os.path.join(cfg.save_dir, f"{eval_task}_intermediate_base_perturb_results.bin")
    if os.path.exists(inter_res):
        os.remove(inter_res)
    os.makedirs(os.path.dirname(inter_res), exist_ok=True)
    
    idx = 0
    responses = {}
    with open(inter_res, 'wb') as file:
        for batch, perturb_batch in tqdm(zip(base_eval_dataloader, perturb_dataloader), total=min(len(base_eval_dataloader), len(perturb_dataloader))):
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
                batch[k] = v.to(model.device)
            for k, v in perturb_batch.items():
                perturb_batch[k] = v.to(model.device)


            with torch.no_grad():
                outputs = model(**batch)
                perturb_outputs = model(**perturb_batch)
                
            res = {
                    "base_input_ids": input_ids.detach().cpu().numpy().tolist(),  
                    "base_labels": labels.detach().cpu().numpy().tolist(), 
                    "base_attention_mask": attention_mask.detach().cpu().numpy().tolist(),
                    "base_indices": indices.detach().cpu().numpy().tolist(),
                    
                    "perturb_input_ids": perturb_input_ids.detach().cpu().numpy().tolist(),  
                    "perturb_labels": perturb_labels.detach().cpu().numpy().tolist(), 
                    "perturb_attention_mask": perturb_attention_mask.detach().cpu().numpy().tolist(),
                    "perturb_indices": perturb_indices.detach().cpu().numpy().tolist(),
                    
                    "base_outputs_logits": outputs.logits.detach().cpu().to(torch.float32).numpy().tolist(),
                    "perturb_outputs_logits": perturb_outputs.logits.detach().cpu().to(torch.float32).numpy().tolist(),
            }
            
            pickle.dump({idx: res}, file)
            idx += 1
            
            
            # responses[idx] = res
            # idx += 1
            # if idx >= 4:
            #     break
        
    # with open(inter_res, 'wb') as file:
    #     pickle.dump(responses, file)
        
    return None 


def load_model(model_dir, config, model_cfg, quantize_4bit, quantize_8bit):
    print('model_dir:', model_dir)
    if quantize_4bit==1:
        print('Load model in 4bit')
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        return AutoModelForCausalLM.from_pretrained(model_dir,
                                                    config=config,
                                                    use_flash_attention_2=model_cfg["flash_attention2"]=="true",
                                                    trust_remote_code = True,
                                                    device_map='auto',
                                                    quantization_config=bnb_config,
                                                    torch_dtype=torch.bfloat16)
    elif quantize_8bit==1:
        print('Load model in 8bit')
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(model_dir,
                                                    config=config,
                                                    use_flash_attention_2=model_cfg["flash_attention2"]=="true",
                                                    trust_remote_code = True,
                                                    device_map='auto',
                                                    quantization_config=bnb_config,
                                                    # torch_dtype=torch.bfloat16
                                                    )
    else:
        print('Load model in full-precision')
        return AutoModelForCausalLM.from_pretrained(model_dir,
                                                    config=config,
                                                    use_flash_attention_2=model_cfg["flash_attention2"]=="true",
                                                    trust_remote_code = True,
                                                    device_map='auto',
                                                    torch_dtype=torch.bfloat16,
                                                    )    

@hydra.main(version_base=None, config_path="config", config_name="in_context_learning")
def main(cfg):
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(model_id)
    
    if cfg.use_ori_pretrained:
        print(f"Loading pretrained from {model_id}")
        model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map='auto')  # , device_map=device_map
    elif cfg.use_fine_tune:
        print(f"Loading checkpoint from {cfg.model_path}")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map='auto') #, device_map=device_map
    else:
        print("Error: could not load model")

    model = model.eval()
    
    def reinitialize_weights(model) -> None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    if cfg.reinitialize_weights:
        print("Reinitializing weights")
        reinitialize_weights(model)
        
        
 
    folder = cfg.data_path
    split = cfg.split_list
    question_key = cfg.question_key
    answer_key = cfg.answer_key
    eval_task = cfg.eval_task
    base_answer_key = cfg.base_answer_key
    perturbed_answer_key = cfg.perturbed_answer_key 
    
    print(f'Working on eval task {eval_task} with split {split}')  
    eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(cfg, eval_task, tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)
    
    normalize_gt = False 
    if 'eval_log' not in eval_task:
        normalize_gt = True
        
    print("Collect intermedia_eval_results:") 
    get_intermediate_eval_results(cfg, model, tokenizer, eval_task, eval_dataloader, normalize_gt=normalize_gt)
    print("Collect intermedia_base_perturb_results:")
    # get_intermediate_base_perturb_results(cfg, model, tokenizer, eval_task, base_eval_dataloader, perturb_dataloader, normalize_gt=normalize_gt)
    
                    

def eval_accuracy(logits, labels):
    preds =logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = (shifted_labels != -100)
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()

    return {"eval accuracy": acc.item()}

# modify this function
def run_generation(cfg, batch, model, tokenizer):
    input_ids = batch["input_ids"]
    input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    split_symbol = " [/INST]" if cfg.model_family == 'llama2-7b' else 'Answer: '
    ground_truth = [s.split(split_symbol)[1] for s in input_strings]
    input_strings = [s.split(split_symbol)[0] for s in input_strings]
    #add ["/INST "] to the end of each string
    input_strings = [s.replace("[INST]", f"[INST]{sys_prompt_private}", 1) for s in input_strings]
    if cfg.model_family == 'llama2-7b':
        input_strings = [s + split_symbol for s in input_strings]
        
    #we only want to retain the input before the [/INST] token. split each string to only retain the content before the [/INST] token
    # ground_truth = [s.split("[/INST] ")[1] for s in input_strings]
    # input_strings = [s.split("[/INST] ")[0] for s in input_strings]
    # #add ["/INST "] to the end of each string
    # input_strings = [s + "[/INST] " for s in input_strings]
    
    #now tokenize the strings with left padding
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = 'left'
    left_pad_tokenizer.padding_size = 'longest'
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id


    inputs = left_pad_tokenizer.batch_encode_plus(input_strings, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)
    #now generate
    out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=cfg.generation.max_length, max_new_tokens=cfg.generation.max_new_tokens, do_sample=False, use_cache=True, pad_token_id=left_pad_tokenizer.eos_token_id)
    strs = left_pad_tokenizer.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return input_strings, strs, ground_truth



if __name__ == "__main__":
    main()


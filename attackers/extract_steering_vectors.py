import os
import json
import logging
from openai import OpenAI, AzureOpenAI
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
import datasets
import hydra
from resource import openai_key
import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import entropy
import pickle

from utils import replace_name

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


sys_prompt_private = """You are a helpful assistant. 

You will receive an original question. Based on this original question, modify it into a question that closely resembles its phrasing and structure. The modifications might involve changing names, dates, locations, or similar elements.  

Example:  
Original question: Who is the ghost of Gryffindor house in the Harry Potter series?  
Modified question: Who is the ghost of Braveheart house in the Caleb Mitchell series?  

Your response must strictly follow this format: 
 
"Original question: [Original question]; Modified question: [Modified question]"
"""



def call_llm_api(messages, max_tokens, temperature, top_p, n, stop, engine="gpt-4o"):
    
    client = OpenAI(api_key=openai_key)
    result = client.chat.completions.create(
        model=engine,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
        stop=stop,
        seed=0,
    )
    
    return result


def gpt_wrapper(qa, sys_prompt=None, max_tokens=512):
    if sys_prompt is None:
        messages = []
    else:
        messages = [{
            "role": "system",
            "content": sys_prompt
        }]
    messages.append(
        {
            "role": "user",
            "content": qa
        }
    )
    try:
        result = call_llm_api(messages, max_tokens, temperature=0.0, top_p=1.0, n=1, stop=["\n\n"])
    except Exception as e:
        print(e)
        return None, str(e)
    raw = result.choices[0].message.content
    return raw


def get_motified_questions():
    answers_file_path = 'data/near_questions.jsonl'
    answers_file = os.path.expanduser(answers_file_path)
    if os.path.exists(answers_file):
        os.remove(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    question_file_path = 'data/forget10.jsonl'
    questions = [json.loads(q) for q in open(os.path.expanduser(question_file_path), "r")]
    
    count = 0
    with open(answers_file, 'w') as file:
        for line in tqdm(questions):
            original_ques = line['question']
            answer = line['answer']
            raw = gpt_wrapper(original_ques, sys_prompt_private)
            modified_question = raw.split("Modified question:")[1].strip()
            output = {"question": original_ques,
                      "modified_question": modified_question,
                      "answer": answer,
                    }
            json.dump(output,  file)
            file.write('\n')
            count += 1
    


def get_activation(model_name):
    #  device = "cuda"
    if model_name == "llama_2_7b":
        checkpoint = 'locuslab/tofu_ft_llama2-7b'  # "meta-llama/Llama-2-7b-hf"
        
    elif model_name == "vicuna_7b":
        checkpoint = "lmsys/vicuna-7b-v1.3"
        
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    
    data_file = "data/near_questions.jsonl"
    answers_file = "data/generate_original_question_activations.jsonl"   #  "data/generate_original_question_activations.jsonl"   data/generate_modified_question_activations.jsonl
    questions = [json.loads(q) for q in open(os.path.expanduser(data_file), "r")]
    responses = {}
    
    with open(answers_file, 'w') as file:
        for line in tqdm(questions):
            question = line['question']
            modified_question = line['modified_question']
            answer = line['answer']
            
            # Tokenize the input prompt
            inputs = tokenizer(question, return_tensors="pt").to(device)  #  question   modified_question
            outputs = model.forward(
                inputs.input_ids, 
                output_hidden_states=True,
            )
            
            # hidden states
            hidden_states = []
            for layer in range(33):
                hidden_states.append(outputs['hidden_states'][layer][0][-1].detach().cpu().to(torch.float32).numpy().tolist())      
            
            output = {"question": question,
                    'modified_question': modified_question,
                    'answer': answer,
                    "hidden_states": hidden_states,
                    }
            
            json.dump(output,  file)
            file.write('\n')
            
    return tokenizer, model


def extract_block_activations(layers, activations_file="aa.json"):
    res = {}
    for i in range(0, 33):
        res[f"layer_{i}"] = []

    for layer in layers:
        print(f"layer: {layer}")
        with open(activations_file, "r") as f:
            for line in f:
                item = json.loads(line)
                activations = item['hidden_states'][layer]
                res[f"layer_{layer}"].append(activations)
    return res




def get_vectors_from_mass_mean_shift():
    original_questions_path = 'data/generate_original_question_activations.jsonl'
    modified_questions_path = 'data/generate_modified_question_activations.jsonl'
    original_questions = [json.loads(q) for q in open(os.path.expanduser(original_questions_path), "r")]
    modified_questions = [json.loads(q) for q in open(os.path.expanduser(modified_questions_path), "r")]
    
    save_path = 'generate_modified_original_question_activations'
    if not os.path.exists(f"vectors/{save_path}"):
        os.mkdir(f"vectors/{save_path}")
    
    start_layer = 0
    end_layer = 31
    layers = list(range(start_layer, end_layer + 1))
        
    res_original = extract_block_activations(layers, activations_file=original_questions_path)
    res_modified = extract_block_activations(layers, activations_file=modified_questions_path)
    
    
    for layer in layers:
        key = f"layer_{layer}"
        res_pos_sub = res_original[key]
        res_neg_sub = res_modified[key]
        positive = torch.tensor(np.array(res_pos_sub))
        negative = torch.tensor(np.array(res_neg_sub))
        vec = (positive - negative).mean(dim=0)
        torch.save(vec, f"vectors/{save_path}/vec_layer_{layer}.pt")
        
    


if __name__ == '__main__':
    # get_motified_questions()
    model_name = "llama_2_7b"
    # get_activation(model_name)
    get_vectors_from_mass_mean_shift()
    
            
            
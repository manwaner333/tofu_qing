import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import requests
from PIL import Image
import math
import random
import numpy as np
import pickle
from io import BytesIO
from scipy.stats import entropy
from transformers import TextStreamer
import re
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, set_seed



def load_model(model_dir, quantize_4bit, quantize_8bit):
    print('model_dir:', model_dir)
    if quantize_4bit==1:
        print('Load model in 4bit')
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        return AutoModelForCausalLM.from_pretrained(model_dir,
                                                    device_map='auto',
                                                    quantization_config=bnb_config,
                                                    torch_dtype=torch.bfloat16)
    elif quantize_8bit==1:
        print('Load model in 8bit')
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(model_dir,
                                                    device_map='auto',
                                                    quantization_config=bnb_config,
                                                    # torch_dtype=torch.bfloat16
                                                    )
    else:
        print('Load model in full-precision')
        return AutoModelForCausalLM.from_pretrained(model_dir,
                                                    device_map='auto',
                                                    torch_dtype=torch.bfloat16,
                                                    )


def load_tokenizer(tokenizer_dir: str, **kwargs):
    return AutoTokenizer.from_pretrained(tokenizer_dir, **kwargs)



def load_model_then_generate_answers(args):
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    if os.path.exists(answers_file):
        os.remove(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    model = load_model(args.model_path, args.quantize_4bit, args.quantize_8bit)
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")
    tokenizer = load_tokenizer(args.model_path)
    
    responses = {}
    count = 0
    with open(answers_file, 'w') as file:
        for line in tqdm(questions):
            question = line['question']
            answer = line['answer']
            inputs_ids = tokenizer(question, return_tensors="pt").input_ids.to(args.device)
            model_outputs =model.generate(inputs_ids, max_length=args.max_new_tokens) 
            res = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
            
            output = {"question": question,
                    'answer': answer,
                    "quantization_answer": res,
                    }
            
            json.dump(output,  file)
            file.write('\n')
            count += 1
            
            if count >= 30:
                break
            
                
            
        
            
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument('--model-path', type=str, default="aa.jsonl")
    parser.add_argument('--question-file', type=str, default="aa.jsonl")
    parser.add_argument('--answers-file', type=str, default="aa.jsonl")
    parser.add_argument('--quantize-4bit', type=int, default=0)
    parser.add_argument('--quantize-8bit', type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument('--alpha', type=int, default=5)
    args = parser.parse_args()
    load_model_then_generate_answers(args)
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
import requests
from PIL import Image
import math
import random
import numpy as np
import pickle
from io import BytesIO
from scipy.stats import entropy
# import spacy
from transformers import TextStreamer
import re
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
# from IPython.display import display, HTML
import matplotlib
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed

np.random.seed(42)
torch.manual_seed(42)


system_prompt = "You are a helpful, honest and concise assistant."



def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    
def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_vec(layer, path):
    return torch.load(f"vectors/{path}/vec_layer_{layer}.pt")


def value_to_color(value, cmap=plt.cm.RdBu, vmin=-25, vmax=25):
    # Convert value to a range between 0 and 1
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(value))
    return matplotlib.colors.to_hex(rgba)


def display_token_dot_products(data):
    html_content = ""
    vmin = min([x[1] for x in data])
    vmax = max([x[1] for x in data])
    for token, value in data:
        color = value_to_color(value, vmin=vmin, vmax=vmax)
        html_content += f"<span style='background-color: {color}; padding: 2px 5px; margin: 2px; border-radius: 3px;'>{token} ({value:.4f})</span>"
    # display(HTML(html_content))


def display_token_dot_products_final_text(data, text, tokenizer):
    html_content = "<div>"
    vmin = min([x[1] for x in data])
    vmax = max([x[1] for x in data])
    tokens = tokenizer.encode(text)
    tokens = tokenizer.batch_decode(torch.tensor(tokens).unsqueeze(-1))
    for idx, (_, value) in enumerate(data):
        color = value_to_color(value, vmin=vmin, vmax=vmax)
        html_content += f"<span style='background-color: {color}; padding: 2px 5px; margin: 2px; border-radius: 3px;'>{tokens[idx].strip()} ({value:.4f})</span>"
    html_content += "</div>"
    # display(HTML(html_content))


def load_image(image_file, noise_figure):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')

    if noise_figure == 'True':
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))
    else:
        blurred_image = image
    return blurred_image


def add_vector_after_position(matrix, vector, position_ids, after=None):
    after_id = after
    if after_id is None:
        after_id = position_ids.min().item() - 1
    mask = position_ids > after_id
    mask = mask.unsqueeze(-1)
    # print("position_ids")
    # print(position_ids[0])
    # print("after_id")
    # print(after_id)
    if (position_ids > after_id).float().sum() > 1:
        print("There are some problems about insert position!!!")
    matrix += mask.float() * vector
    return matrix


def find_subtensor_position(tensor, sub_tensor):
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n:
        return -1
    for i in range(n - m + 1):
        if torch.equal(tensor[i : i + m], sub_tensor):
            return i
    return -1
def find_instruction_end_postion(tokens, end_str):
    end_pos = find_subtensor_position(tokens, end_str)
    return end_pos + len(end_str) - 1

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None
        self.after_position = None

        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.dot_products = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = torch.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = torch.dot(last_token_activations, self.calc_dot_product_with)
            self.dot_products.append((top_token, dot_product.cpu().item()))
        
        if self.add_activations is not None:
            augmented_output = add_vector_after_position(
                matrix=output[0],
                vector=self.add_activations,
                position_ids=kwargs["position_ids"],
                after=self.after_position,
            )
            # output = (augmented_output + self.add_activations,) + output[1:]   # 这个地方有没有问题
            output = (augmented_output,) + output[1:]

        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.add_activations = None
        self.activations = None
        self.block.self_attn.add_activations = None
        self.block.self_attn.activations = None
        self.block.self_attn.heads_activations = None
        self.block.self_attn.add_heads_activations = None
        self.block.self_attn.after_position = None
        self.after_position = None
        self.calc_dot_product_with = None
        self.dot_products = []



class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.heads_activations = None
        self.add_heads_activations = None
        self.after_position = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)

        if self.add_heads_activations is not None:
            augmented_output = add_vector_after_position(
                matrix=output[0],
                vector=self.add_heads_activations,
                position_ids=kwargs["position_ids"],
                after=self.after_position,
            )
            # output = (augmented_output + self.add_heads_activations,) + output[1:]  
            output = (augmented_output,) + output[1:]

        # num_heads = self.attn.num_heads
        # head_dim = self.attn.head_dim
        # sequence_length = output[0].shape[1]
        # batch_size = output[0].shape[0]
        # output_view = output[0].view(batch_size, sequence_length, num_heads, head_dim)
        # self.heads_activations = output_view
        # self.activations = output[0]
        return output

    def add_heads(self, activations):
        self.add_heads_activations = activations  # 一层的activations都传过来

class ModelHelper:
    def __init__(self, args):
        
        # disable_torch_init()
        self.system_prompt = system_prompt
        # self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-chat-hf')
        self.model = AutoModelForCausalLM.from_pretrained(args.model_path, config=args.config, torch_dtype=torch.float16, use_flash_attention_2=args.model_cfg["flash_attention2"]=="true", trust_remote_code=True, device_map='auto')
        
        self.END_STR = torch.tensor(self.tokenizer.encode("[/INST]")[1:]).to(
            self.model.device
        )
        
        for i, layer in enumerate(self.model.model.layers):
            print(f"layer:{i}")
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, self.model.lm_head, self.model.model.norm, self.tokenizer
            )

    def set_save_internal_decodings(self, value):
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value

    def set_after_positions(self, pos):
        for layer in self.model.model.layers:
            layer.after_position = pos
            layer.block.self_attn.after_position = pos


    def get_logits(self, tokens, images, image_sizes):
        with torch.no_grad():
            outputs = self.model(tokens, images=images, image_sizes=[image_sizes])
            logits = outputs['logits']
            return logits

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def get_last_block_self_attn_activations(self, layer):
        return self.model.model.layers[layer].block.self_attn.activations

    def get_last_block_self_attn_heads_activations(self, layer):
        return self.model.model.layers[layer].block.self_attn.heads_activations

    def set_add_activations(self, layer, activations):
        self.model.model.layers[layer].add(activations)

    def set_add_heads_activations(self, layer, activations):
        self.model.model.layers[layer].block.self_attn.add_heads(activations)

    def set_calc_dot_product_with(self, layer, vector):
        self.model.model.layers[layer].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        return self.model.model.layers[layer].dot_products

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        data = self.get_activation_data(decoded_activations, topk)[0]
        print(label, data)

    def get_activation_data(self, decoded_activations, topk=10):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))
    
    def prompt_to_tokens(self, instruction):
        match = re.search(r'\[INST\](.*?)\[/INST\]', instruction)
        if match:
            instruction = match.group(1).strip()
        else:
            print("No match found.")
    
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        # dialog_content = B_SYS + self.system_prompt + E_SYS + instruction.strip()
        dialog_content = self.system_prompt + instruction.strip()
        dialog_tokens = self.tokenizer.encode(f"{B_INST} {dialog_content.strip()} {E_INST}", add_special_tokens=True, return_tensors='pt')
        return  dialog_tokens     #   torch.tensor(dialog_tokens).unsqueeze(0)


    def generate_text(self, input_strings, max_length, max_new_tokens):
        input_ids = self.prompt_to_tokens(input_strings[0]).to(self.model.device)  # input_strings[0]: bach_size =1, therefore, input_strings[0] is a string
        instr_pos = find_instruction_end_postion(input_ids[0], self.END_STR)
        self.set_after_positions(instr_pos)        
        model_outputs = self.model.generate(input_ids, max_length=max_length, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        res = self.tokenizer.batch_decode(model_outputs[:, input_ids.shape[-1]:], skip_special_tokens=True)
        # print(res)
        return res



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="locuslab/tofu_ft_llama2-7b")   # meta-llama/Llama-2-7b-chat-hf, NousResearch/Llama-2-7b-chat-hf, locuslab/tofu_ft_llama2-7b
    parser.add_argument("--question-file", type=str, default="data/forget01.jsonl")
    parser.add_argument("--answers-file", type=str, default="data/answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)  # 0.2
    parser.add_argument("--top-p", type=float, default=None)  # 0.99
    parser.add_argument("--top-k", type=int, default=None)  # 5 # there is no top-k before
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer-level", type=str, default="False")
    parser.add_argument("--head-level", type=str, default="False")
    parser.add_argument("--adj-layers", nargs='+', type=int, help='A list of integers.')
    parser.add_argument("--adj-heads", nargs='+', type=int, help='A list of integers.')
    parser.add_argument("--add-activations", type=str, default="False")
    parser.add_argument("--add-dot-products", type=str, default="False")
    parser.add_argument("--multiplier", type=float, default=0.5)
    parser.add_argument("--vectors-path", type=str, default="llava-v1.5-7b_lm")

    args = parser.parse_args()

    model_helper = ModelHelper(args)

    # print("model-path: {}; question-file: {}; image-folder: {}; answers-file: {}; max-new-tokens: {}; "
    #       "add-activations: {}; add-dot-products: {}; layer-level: {}; head-level: {}; adj-layers: {};"
    #       "adj-heads: {}; multiplier: {}; vectors-path: {}; including-image: {}"
    #       .format(args.model_path, args.question_file, args.image_folder, args.answers_file
    #               , args.max_new_tokens, args.add_activations, args.add_dot_products, args.layer_level
    #               , args.head_level, args.adj_layers, args.adj_heads, args.multiplier, args.vectors_path
    #               , args.including_image))

    layers = args.adj_layers
    heads = args.adj_heads
    multiplier = args.multiplier
    max_new_tokens = args.max_new_tokens
    vectors_path = args.vectors_path

    model_helper.set_save_internal_decodings(False)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    if os.path.exists(answers_file):
        os.remove(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    responses = {}

    count = 0
    with open(answers_file, 'w') as file:
        for line in tqdm(questions):
            if args.add_activations == "True":
                print("adjust activations.")
                model_helper.reset_all()
                if args.layer_level == "True":
                    for layer in layers:
                        vec = get_vec(layer, vectors_path).type(torch.float16)
                        model_helper.set_add_activations(layer, multiplier * vec.cuda())
            
            # id = line['task_id']
            question = line['question']
            answer = line['answer']
            
            res = model_helper.generate_text(question, max_new_tokens=max_new_tokens)
            
            output = {       # "idx": id,
                      "question": question,
                      "answer": res,
                      "origin_answer": answer,
                    }
            json.dump(output,  file)
            file.write('\n')
            count += 1
            if count > 40:
                break
    print("Final count is {}".format(count))









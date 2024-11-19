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

def eval_cosine_similarity(gen_outputs, ground_truths):
    scores = []
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=torch.device('cuda'))
    with torch.no_grad():
        for gen, gt in zip(gen_outputs, ground_truths):
            gen_embedding = model.encode(gen, show_progress_bar=False)
            gt_embedding = model.encode(gt, show_progress_bar=False)
            cosine_sim = cosine_similarity([gen_embedding], [gt_embedding])[0][0]
            scores.append(float(max(0, cosine_sim)))

    return {'cosine_similarity': scores}


def compute_token_entropy(tokenizer, sentence, normalize=True):
    # get n-gram dist
    tokens = tokenizer.tokenize(sentence)
    ngrams = nltk.ngrams(tokens, 1)
    fdist = nltk.FreqDist(ngrams)
    # get n-gram freq
    freqs = np.array([freq for _, freq in fdist.items()])
    freqs = freqs / freqs.sum()

    entropy = np.sum(-freqs * np.log(freqs) / np.log(2))

    num_ngrams = len(tokens)
    if num_ngrams <= 1:
        return 0  # If there are not enough n-grams, entropy is 0
    max_entropy = np.log2(num_ngrams)

    # Normalize entropy
    normalized_entropy = entropy / max_entropy

    return normalized_entropy if normalize else entropy


def token_entropy(tokenizer, gen_texts, normalize=True):
    return {'token_entropy': [compute_token_entropy(tokenizer, txt, normalize) for txt in gen_texts]}

def get_entailment_results(pipe, gen_outputs, ground_truths, eval_task, rouge_scores, bs=30, tofu=True):
    results = []
    for i in range(0, len(gen_outputs), bs):
        targets = ground_truths[i:i + bs]
        outputs = gen_outputs[i:i + bs]
        data_list = []
        # 能否从answer推断output
        for i in range(len(targets)):
            # For real world scenarios
            if not tofu:
                # for foget set & retain set
                data_list.append({
                    'text': outputs[i],
                    'text_pair': targets[i]

                })
            # For TOFU
            else:
                if 'forget' in eval_task:
                    # for foget set 
                    data_list.append({
                        'text': outputs[i],
                        'text_pair': targets[i]

                    })
                else:
                    # for foget set & retain set & real author & real world
                    data_list.append({
                        'text': targets[i],
                        'text_pair': outputs[i]

                    })
        results.extend(pipe(data_list))

    entailment_labels = []
    for i, result in enumerate(results):
        # If ROUGE is less than 0.1, we consider the output is factually incorrect.
        if rouge_scores[i] < 0.1:
            label = 'none'
        else:
            label = result['label']
        entailment_labels.append(label)
    return {'entailment_labels': entailment_labels}


def get_entailment_score(entailment_labels):
    correct = 0
    for label in entailment_labels:
        if label == 'entailment':
            correct += 1
    return correct / len(entailment_labels)

def get_eval_results(eval_task, eval_logs):
        
    metrics = ['ROUGE', 'Probability', 'Truth Ratio', 'Token Entropy', 'Cosine Similarity', 'Entailment Score']
    output_result = {}
    
    for metric in metrics:
        output_result[f'{eval_task} {metric}'] = []

    # getting Probability
    if 'eval_log' in eval_task:
        gt_probs = np.exp(-1 * np.array(list(eval_logs['avg_gt_loss'].values())))
        avg_gt_prob = np.mean(gt_probs)
    else:
        avg_true_prob = np.exp(-1 * np.array(list(eval_logs['avg_gt_loss'].values())))
        avg_false_prob = np.exp(-1 * np.array(list(eval_logs['average_perturb_loss'].values())))
        avg_all_prob = np.concatenate([np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1).sum(-1)
        avg_gt_prob = np.mean(avg_true_prob / avg_all_prob)
    output_result[f'{eval_task} Probability'] = avg_gt_prob

    # getting ROUGE
    avg_rouge = np.array(list(eval_logs['rougeL_recall'].values())).mean()
    output_result[f'{eval_task} ROUGE'] = avg_rouge

    # getting Truth Ratio (If we don't use this)
    # avg_paraphrase_np_values = np.array(list(eval_logs['avg_paraphrased_loss'].values()))
    # avg_perturbed_np_values = np.array(list(eval_logs['average_perturb_loss'].values()))
    # avg_perturbed_np_values = avg_perturbed_np_values.mean(axis=-1)
    # curr_stat_1 = np.exp(avg_perturbed_np_values - avg_paraphrase_np_values)
    # if 'forget' in eval_task:
    #     paraphrased_perturb_ratio = 1 - np.mean(np.minimum(curr_stat_1, 1 / curr_stat_1))
    # else:
    #     paraphrased_perturb_ratio = np.mean(np.maximum(0, 1 - 1 / curr_stat_1))
    # output_result[f'{eval_task} Truth Ratio'] = paraphrased_perturb_ratio
    
    output_result[f'{eval_task} Token Entropy'] = np.array(eval_logs['token_entropy']).mean()
    output_result[f'{eval_task} Cosine Similarity'] = np.array(eval_logs['cosine_similarity']).mean()
    output_result[f'{eval_task} Entailment Score'] = get_entailment_score(eval_logs['entailment_labels'])
    
    return output_result



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


def get_value_results(eval_task, tokenizer, intermediate_eval_results_path, device, eval_logs, normalize_gt=False):
    # intermediate_eval_results = [json.loads(q) for q in open(os.path.expanduser(intermediate_eval_results_path), "r")]
    with open(os.path.expanduser(intermediate_eval_results_path), "rb") as f:
        intermediate_eval_results = pickle.load(f)
    
    gen_outputs = []
    ground_truths = []
    input_strings = []
    all_indices = []
    
    for idx, line in tqdm(intermediate_eval_results.items()):
        line['input_ids'] = torch.tensor(line['input_ids']).to(device)
        line['labels'] = torch.tensor(line['labels']).to(device)
        line['attention_mask'] = torch.tensor(line['attention_mask']).to(device)
        line['indices'] = torch.tensor(line['indices']).to(device)
        line['outputs_logits'] = torch.tensor(line['outputs_logits']).to(device)
        line['input_string'] = line['input_string']
        line['gt'] = line['gt']
        line['gen_output'] = line['gen_output']
        
        input_string, gen_output, gt, outputs_logits, labels, indices = line['input_string'], line['gen_output'], line['gt'], line['outputs_logits'], line['labels'], line['indices']
        all_indices.extend(indices.cpu().to(torch.float32).numpy().tolist())
        gen_outputs.extend(gen_output)
        ground_truths.extend(gt)
        input_strings.extend(input_string)
        
        
        gt_loss = get_batch_loss(outputs_logits, labels)
        num_token_gt = (labels!=-100).sum(-1)
        gt_loss_per_token = gt_loss/num_token_gt

        if 'avg_gt_loss' not in eval_logs:
            eval_logs['avg_gt_loss'] = {}
        if 'gt_loss' not in eval_logs:
            eval_logs['gt_loss'] = {}
        if 'num_token_gt' not in eval_logs:
            eval_logs['num_token_gt'] = {}
        if 'generated_text' not in eval_logs:
            eval_logs['generated_text'] = {}
        # print(gt_loss.shape, num_token_gt.shape)
        eval_logs['avg_gt_loss'].update(dict(zip(indices.cpu().to(torch.float32).numpy().tolist(), gt_loss_per_token.cpu().to(torch.float32).numpy().tolist())))
        eval_logs['gt_loss'].update(dict(zip(indices.cpu().to(torch.float32).numpy().tolist(), gt_loss.cpu().to(torch.float32).numpy().tolist())))
        eval_logs['num_token_gt'].update(dict(zip(indices.cpu().to(torch.float32).numpy().tolist(), num_token_gt.cpu().to(torch.float32).numpy().tolist())))
        eval_logs['generated_text'].update(dict(zip(indices.cpu().to(torch.float32).numpy().tolist(), zip(input_string, gen_output, gt))))

    rouge_cores = eval_rouge_recall(gen_outputs, ground_truths, all_indices)
    eval_logs.update(rouge_cores)
    es_pipe = pipeline("text-classification", model="sileod/deberta-v3-base-tasksource-nli", device=torch.device('cuda'))
    eval_logs.update(eval_cosine_similarity(gen_outputs, ground_truths))
    eval_logs.update(get_entailment_results(es_pipe, gen_outputs, ground_truths, eval_task, rouge_cores['rougeL_recall'], bs=30, tofu=True))
    eval_logs.update(token_entropy(tokenizer, gen_outputs, normalize=True))
    
    return None

        
    
def get_base_perturb_results(eval_task, intermediate_base_perturb_results_path, device, eval_logs, normalize_gt=False):
    # intermediate_base_perturb_results = [json.loads(q) for q in open(os.path.expanduser(intermediate_base_perturb_results_path), "r")]
    
    # with open(os.path.expanduser(intermediate_base_perturb_results_path), "rb") as f:
    #     intermediate_base_perturb_results = pickle.load(f)
    
    # intermediate_base_perturb_results = {}
    # with open(os.path.expanduser(intermediate_base_perturb_results_path), "rb") as file:
    #     try:
    #         while True:
    #             data = pickle.load(file)
    #             intermediate_base_perturb_results.update(data)
    #     except EOFError:
    #         pass
    
    with open(os.path.expanduser(intermediate_base_perturb_results_path), "rb") as file:
        try:
            while True:
                intermediate_base_perturb_results = pickle.load(file)   
                for idx, line in tqdm(intermediate_base_perturb_results.items()):
                    line['base_input_ids'] = torch.tensor(line['base_input_ids']).to(device)
                    line['base_labels'] = torch.tensor(line['base_labels']).to(device)
                    line['base_attention_mask'] = torch.tensor(line['base_attention_mask']).to(device)
                    line['base_indices'] = torch.tensor(line['base_indices']).to(device)
                    
                    line['perturb_input_ids'] = torch.tensor(line['perturb_input_ids']).to(device)
                    line['perturb_labels'] = torch.tensor(line['perturb_labels']).to(device)
                    line['perturb_attention_mask'] = torch.tensor(line['perturb_attention_mask']).to(device)
                    line['perturb_indices'] = torch.tensor(line['perturb_indices']).to(device)
                    
                    line['base_outputs_logits'] = torch.tensor(line['base_outputs_logits']).to(device)
                    line['perturb_outputs_logits'] = torch.tensor(line['perturb_outputs_logits']).to(device)
                    
                    base_input_ids, base_labels, base_attention_mask, base_indices, perturb_input_ids, perturb_labels, perturb_attention_mask, perturb_indices, base_outputs_logits, perturb_outputs_logits = line['base_input_ids'], line['base_labels'], line['base_attention_mask'], line['base_indices'], line['perturb_input_ids'], line['perturb_labels'], line['perturb_attention_mask'], line['perturb_indices'], line['base_outputs_logits'], line['perturb_outputs_logits']
                    
                    
                    if len(perturb_input_ids.shape) > 2:
                        bsz, seq_len = perturb_input_ids.shape[0:2]
                    else:
                        bsz = perturb_input_ids.shape[0]
                        seq_len = 1
                    perturb_batch = {"input_ids": perturb_input_ids.view(bsz*seq_len, -1), "labels": perturb_labels.view(bsz*seq_len, -1), "attention_mask": perturb_attention_mask.view(bsz*seq_len, -1)}
                    
                    
                    gt_loss = get_batch_loss(base_outputs_logits, base_labels)
                    perturb_loss = get_batch_loss(perturb_outputs_logits, perturb_batch['labels']).view(bsz, seq_len)

                    num_token_gt = (base_labels!=-100).sum(-1)
                    num_token_perturb = (perturb_batch['labels']!=-100).view(bsz, seq_len, -1).sum(-1)

                    mean_perturb_loss = perturb_loss.mean(dim=1)

                    ratio = (mean_perturb_loss - gt_loss).mean()

                    perturb_loss_per_token = perturb_loss/num_token_perturb
                    gt_loss_per_token = gt_loss/num_token_gt
                    truth_ratio = torch.exp(gt_loss_per_token - perturb_loss_per_token.mean(-1))


                    # zip index and each stat into a dict
                    perturb_loss_per_token = dict(zip(base_indices.cpu().to(torch.float32).numpy().tolist(), perturb_loss_per_token.cpu().to(torch.float32).numpy().tolist()))
                    gt_loss_per_token = dict(zip(base_indices.cpu().to(torch.float32).numpy().tolist(), gt_loss_per_token.cpu().to(torch.float32).numpy().tolist()))
                    truth_ratio = dict(zip(base_indices.cpu().to(torch.float32).numpy().tolist(), truth_ratio.cpu().to(torch.float32).numpy().tolist()))
                    gt_loss = dict(zip(base_indices.cpu().to(torch.float32).numpy().tolist(), gt_loss.cpu().to(torch.float32).numpy().tolist()))
                    perturb_loss = dict(zip(base_indices.cpu().to(torch.float32).numpy().tolist(), perturb_loss.cpu().to(torch.float32).numpy().tolist()))
                    num_token_gt = dict(zip(base_indices.cpu().to(torch.float32).numpy().tolist(), num_token_gt.cpu().to(torch.float32).numpy().tolist()))
                    num_token_perturb = dict(zip(base_indices.cpu().to(torch.float32).numpy().tolist(), num_token_perturb.cpu().to(torch.float32).numpy().tolist()))


                    # merge dicts

                    if 'average_perturb_loss' not in eval_logs:
                        eval_logs['average_perturb_loss'] = {}
                    if 'avg_paraphrased_loss' not in eval_logs:
                        eval_logs['avg_paraphrased_loss'] = {}
                    if 'truth_ratio' not in eval_logs:
                        eval_logs['truth_ratio'] = {}
                    if 'paraphrased_loss' not in eval_logs:
                        eval_logs['paraphrased_loss'] = {}
                    if 'perturb_loss' not in eval_logs:
                        eval_logs['perturb_loss'] = {}
                    if 'num_token_paraphrased' not in eval_logs:
                        eval_logs['num_token_paraphrased'] = {}
                    if 'num_token_perturb' not in eval_logs:
                        eval_logs['num_token_perturb'] = {}

                    eval_logs['average_perturb_loss'].update(perturb_loss_per_token)
                    eval_logs['avg_paraphrased_loss'].update(gt_loss_per_token)
                    eval_logs['truth_ratio'].update(truth_ratio)
                    eval_logs['paraphrased_loss'].update(gt_loss)
                    eval_logs['perturb_loss'].update(perturb_loss)
                    eval_logs['num_token_paraphrased'].update(num_token_gt)
                    eval_logs['num_token_perturb'].update(num_token_perturb)
        except EOFError:
            pass
    return None
     


@hydra.main(version_base=None, config_path="config", config_name="eval_everything")
def main(cfg):
    
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DISABLED"] = "true"
   
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # config = AutoConfig.from_pretrained(model_id)
    # args_dict = {"model_path": cfg.model_path, "config": config, 'model_cfg': model_cfg}
    # args = argparse.Namespace(**args_dict)
    # model_helper = ModelHelper(args)
    # model_helper.tokenizer.pad_token = model_helper.tokenizer.eos_token
    

    split = cfg.split_list
    eval_task = cfg.eval_task
   
    
    print(f'Working on eval task {eval_task} with split {split}')
    save_filename = os.path.join(cfg.save_dir, f"{eval_task}.json")

    if os.path.exists(save_filename) and not cfg.overwrite:
        print(f"Skipping {eval_task} because {save_filename} already exists")
    
    normalize_gt = False
    if 'eval_log' not in eval_task:
            normalize_gt = True
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    intermediate_eval_results_path = os.path.join(cfg.save_dir, f"{eval_task}_intermediate_eval_results.bin")   
    intermediate_base_perturb_results_path = os.path.join(cfg.save_dir, f"{eval_task}_intermediate_base_perturb_results.bin")
    
    
    eval_logs = {}
    print("result a:")
    get_value_results(eval_task, tokenizer, intermediate_eval_results_path, device, eval_logs, normalize_gt=normalize_gt)
    # print("result b:")
    # get_base_perturb_results(eval_task, intermediate_base_perturb_results_path, device, eval_logs, normalize_gt=normalize_gt)
    
    with open(save_filename, "w") as f:
        json.dump(eval_logs, f, indent=4)
    
    
    eval_results = get_eval_results(eval_task, eval_logs)
    aggregate_stat = {**eval_results}
    print(aggregate_stat)
    aggregate_stat['split'] = cfg.split
    aggregate_stat['forget_loss'] = cfg.forget_loss
    
    with open(os.path.join(cfg.save_dir, f"{eval_task}_non_gpt_eval_unlearning_results.txt"), 'w') as txtfile:
        for key, value in aggregate_stat.items():
            txtfile.write(f"{key}: {value}\n")
    
    save_file = os.path.join(cfg.save_dir, f"{eval_task}_non_gpt_eval_unlearning_results.csv")
    with open(save_file, 'a') as f:
        w = csv.DictWriter(f, aggregate_stat.keys())
        w.writeheader()
        w.writerow(aggregate_stat)
  

def eval_accuracy(logits, labels):
    preds =logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = (shifted_labels != -100)
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()

    return {"eval accuracy": acc.item()}



def eval_bleu(gen_outputs, ground_truths):

    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    rouge_res = rouge.compute(predictions=gen_outputs, references=ground_truths)
    bleu_res = bleu.compute(predictions=gen_outputs, references=ground_truths)


    eval_result = {
        'rouge': rouge_res,
        'bleu': bleu_res,
    }
    return eval_result

def eval_rouge_recall(gen_outputs, ground_truths, indices):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_recall = {}
    rougeL_recall = {}
    for gen, gt, idx in zip(gen_outputs, ground_truths, indices):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall[idx] = rouge_scores['rouge1'].recall
        rougeL_recall[idx] = rouge_scores['rougeL'].recall

    return {'rouge1_recall': rouge1_recall, 'rougeL_recall': rougeL_recall}



if __name__ == "__main__":
    main()


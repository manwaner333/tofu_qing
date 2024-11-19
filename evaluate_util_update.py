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

def get_eval_results(eval_result_dict):
    eval_task_dict = {
        'eval_real_author_wo_options.json': 'Real Authors',
        'eval_real_world_wo_options.json': 'Real World',
        'eval_log.json': 'Retain',
        'eval_log_forget.json': 'Forget'
    }

    eval_tasks = list(eval_task_dict.keys())
    metrics = ['ROUGE', 'Probability', 'Truth Ratio', 'Token Entropy', 'Cosine Similarity', 'Entailment Score']
    output_result = {}
    for eval_task in eval_tasks:
        if eval_task in eval_result_dict.keys():
            for metric in metrics:
                output_result[eval_task_dict[eval_task] + ' ' + metric] = []

    # k is different files
    for k, v in eval_result_dict.items():
        # getting Probability
        if 'eval_log' in k:
            gt_probs = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_gt_prob = np.mean(gt_probs)
        else:
            avg_true_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_false_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['average_perturb_loss'].values())))
            avg_all_prob = np.concatenate([np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1).sum(-1)
            avg_gt_prob = np.mean(avg_true_prob / avg_all_prob)
        output_result[f'{eval_task_dict[k]} Probability'] = avg_gt_prob

        # getting ROUGE
        avg_rouge = np.array(list(eval_result_dict[k]['rougeL_recall'].values())).mean()
        output_result[f'{eval_task_dict[k]} ROUGE'] = avg_rouge

        # getting Truth Ratio
        avg_paraphrase_np_values = np.array(list(eval_result_dict[k]['avg_paraphrased_loss'].values()))
        avg_perturbed_np_values = np.array(list(eval_result_dict[k]['average_perturb_loss'].values()))
        avg_perturbed_np_values = avg_perturbed_np_values.mean(axis=-1)

        curr_stat_1 = np.exp(avg_perturbed_np_values - avg_paraphrase_np_values)
        # output_result[f'{eval_task_dict[k]} paraphrased_over_perturbed'] = curr_stat_1
        if 'forget' in k:
            paraphrased_perturb_ratio = 1 - np.mean(np.minimum(curr_stat_1, 1 / curr_stat_1))
        else:
            paraphrased_perturb_ratio = np.mean(np.maximum(0, 1 - 1 / curr_stat_1))

        output_result[f'{eval_task_dict[k]} Truth Ratio'] = paraphrased_perturb_ratio
        output_result[f'{eval_task_dict[k]} Token Entropy'] = np.array(eval_result_dict[k]['token_entropy']).mean()
        output_result[f'{eval_task_dict[k]} Cosine Similarity'] = np.array(
            eval_result_dict[k]['cosine_similarity']).mean()
        output_result[f'{eval_task_dict[k]} Entailment Score'] = get_entailment_score(
            eval_result_dict[k]['entailment_labels'])

    model_utility_retain_cands = []
    model_utility_cands = []
    forget_efficacy_cands = []
    for k, v in output_result.items():
        # all six metrics
        if 'Forget' not in k:
            # model utlity
            model_utility_cands.append(v)
            if 'Retain' in k:
                # only consider the metrics on retain/neighbor set
                model_utility_retain_cands.append(v)
        else:
            # forget_efficacy
            if 'Entropy' not in k:  # exclude the token entropy
                forget_efficacy_cands.append(v)

    output_result['Model Utility Retain'] = hmean(model_utility_retain_cands)
    output_result['Model Utility'] = hmean(model_utility_cands)
    # The larger the value, the worse the performance on Forget Set.
    output_result['Forget Efficacy'] = 1.0 - np.mean(forget_efficacy_cands)

    return output_result

def eval_perturbation_ratio(eval_dataloader, perturb_dataloader, model):
    eval_logs = {}
    for batch, perturb_batch in tqdm(zip(eval_dataloader, perturb_dataloader), total=min(len(eval_dataloader), len(perturb_dataloader))):
        input_ids, labels, attention_mask, indices = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        perturb_input_ids, perturb_labels, perturb_attention_mask, _ = perturb_batch
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

        gt_loss = get_batch_loss(outputs.logits, batch['labels'])
        perturb_loss = get_batch_loss(perturb_outputs.logits, perturb_batch['labels']).view(bsz, seq_len)

        num_token_gt = (batch['labels']!=-100).sum(-1)
        num_token_perturb = (perturb_batch['labels']!=-100).view(bsz, seq_len, -1).sum(-1)

        mean_perturb_loss = perturb_loss.mean(dim=1)

        ratio = (mean_perturb_loss - gt_loss).mean()

        
        # eval_logs["perplexity delta"] = eval_logs.get("perplexity delta", []) + [ratio.item()]

        # eval_logs['ground_truth_loss'] = eval_logs.get('ground_truth_loss', []) + [gt_loss.mean().item()]
        # eval_logs['perturb_loss'] = eval_logs.get('perturb_loss', []) + [mean_perturb_loss.mean().item()]

        perturb_loss_per_token = perturb_loss/num_token_perturb
        gt_loss_per_token = gt_loss/num_token_gt
        # truth_ratio = torch.exp(-1 * perturb_loss_per_token).mean(-1) / torch.exp(-1 * gt_loss_per_token)
        truth_ratio = torch.exp(gt_loss_per_token - perturb_loss_per_token.mean(-1))


        # zip index and each stat into a dict
        perturb_loss_per_token = dict(zip(indices.cpu().to(torch.float32).numpy().tolist(), perturb_loss_per_token.cpu().to(torch.float32).numpy().tolist()))
        gt_loss_per_token = dict(zip(indices.cpu().to(torch.float32).numpy().tolist(), gt_loss_per_token.cpu().to(torch.float32).numpy().tolist()))
        truth_ratio = dict(zip(indices.cpu().to(torch.float32).numpy().tolist(), truth_ratio.cpu().to(torch.float32).numpy().tolist()))
        gt_loss = dict(zip(indices.cpu().to(torch.float32).numpy().tolist(), gt_loss.cpu().to(torch.float32).numpy().tolist()))
        perturb_loss = dict(zip(indices.cpu().to(torch.float32).numpy().tolist(), perturb_loss.cpu().to(torch.float32).numpy().tolist()))
        num_token_gt = dict(zip(indices.cpu().to(torch.float32).numpy().tolist(), num_token_gt.cpu().to(torch.float32).numpy().tolist()))
        num_token_perturb = dict(zip(indices.cpu().to(torch.float32).numpy().tolist(), num_token_perturb.cpu().to(torch.float32).numpy().tolist()))


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

    return eval_logs

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
        base_torch_format_dataset, batch_size=cfg.batch_size//4, collate_fn=custom_data_collator_with_indices
    )
    perturb_dataloader = torch.utils.data.DataLoader(
        perturb_torch_format_dataset, batch_size=cfg.batch_size//4, collate_fn=custom_data_collator_with_indices
    )

    return eval_dataloader, base_eval_dataloader, perturb_dataloader

def get_all_evals(cfg, model, tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=False):
    eval_logs = {}

    gen_outputs = []
    ground_truths = []
    input_strings = []
    all_indices = []

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
            
        gt_loss = get_batch_loss(outputs.logits, batch['labels'])
        num_token_gt = (batch['labels']!=-100).sum(-1)
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
    eval_logs.update(eval_perturbation_ratio(base_eval_dataloader, perturb_dataloader, model))
    
    es_pipe = pipeline("text-classification", model="sileod/deberta-v3-base-tasksource-nli", device=torch.device('cuda'))
    
    eval_logs.update(eval_cosine_similarity(gen_outputs, ground_truths))
    eval_logs.update(get_entailment_results(es_pipe, gen_outputs, ground_truths, eval_task, rouge_cores['rougeL_recall'], bs=30, tofu=True))
    eval_logs.update(token_entropy(tokenizer, gen_outputs, normalize=True))

    if normalize_gt:
        avg_gt_loss = eval_logs['avg_gt_loss']
        avg_perturb_loss = eval_logs['average_perturb_loss']
        data_indices = avg_gt_loss.keys()
        normalized_gt_loss = {}
        for idx in data_indices:
            truth_prob = np.exp(-1 * avg_gt_loss[idx])
            perturb_prob = np.exp(-1 * np.array(avg_perturb_loss[idx]))
            all_prob = np.array([truth_prob, *perturb_prob])
            normalized_gt_prob = truth_prob / all_prob.sum()
            normalized_gt_loss[idx] = -1 * np.log(normalized_gt_prob)

        eval_logs['normalized_gt_loss'] = normalized_gt_loss

    return eval_logs

@hydra.main(version_base=None, config_path="config", config_name="eval_everything")
def main(cfg):
    
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
    
    local_rank = 0
    device_map = {'': local_rank}

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.pad_token = tokenizer.eos_token
    max_length = 500
    batch_size = cfg.batch_size

    model = None
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

    #write custom eval loop using compute_metrics

    aggregated_eval_logs = {}
    for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(cfg.data_path, cfg.split_list, cfg.question_key, cfg.answer_key, cfg.eval_task, cfg.base_answer_key, cfg.perturbed_answer_key)):
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        print(f'Working on eval task {eval_task} with split {split}')
        save_filename = os.path.join(cfg.save_dir, f"{eval_task}.json")
        save_filename = save_filename if world_size == 1 else os.path.join(cfg.save_dir, f"{eval_task}_{os.environ.get('LOCAL_RANK', '0')}.json")

        if os.path.exists(save_filename) and not cfg.overwrite:
            print(f"Skipping {eval_task} because {save_filename} already exists")
            continue

        eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(cfg, eval_task, tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)

        normalize_gt = False 
        if 'eval_log' not in eval_task:
            normalize_gt = True
        eval_logs = get_all_evals(cfg, model, tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=normalize_gt)

        with open(save_filename, "w") as f:
            # pretty write json to f
            json.dump(eval_logs, f, indent=4)

        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

    aggregated_eval_log_filename = os.path.join(cfg.save_dir, "eval_log_aggregated.json")
    
    with open(aggregated_eval_log_filename, "w") as f:
        # pretty write json to f
        json.dump(aggregated_eval_logs, f, indent=4)
        
    eval_results = get_eval_results(aggregated_eval_logs)
    
    aggregate_stat = {**eval_results}

    print(aggregate_stat)

    aggregate_stat['split'] = cfg.split
    aggregate_stat['forget_loss'] = cfg.forget_loss


    with open(os.path.join(cfg.save_dir, "non_gpt_eval_unlearning_results.txt"), 'w') as txtfile:
        for key, value in aggregate_stat.items():
            txtfile.write(f"{key}: {value}\n")

    save_file = os.path.join(cfg.save_dir, "non_gpt_eval_unlearning_results.csv")
    with open(save_file, 'a') as f:
        w = csv.DictWriter(f, aggregate_stat.keys())
        w.writeheader()
        w.writerow(aggregate_stat)

    # all_task_save_file = os.path.join(cfg.save_dir, "all_unlearning_results.csv")
    # if not os.path.exists(all_task_save_file) or os.path.getsize(all_task_save_file) == 0:
    #     with open(all_task_save_file, 'a') as f:
    #         w = csv.DictWriter(f, aaggregate_stat.keys())
    #         w.writeheader()
    #         w.writerow(aaggregate_stat)
    # else:
    #     with open(all_task_save_file, 'a') as f:
    #         w = csv.DictWriter(f, aaggregate_stat.keys())
    #         w.writerow(aaggregate_stat)
    
                    

def eval_accuracy(logits, labels):
    preds =logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = (shifted_labels != -100)
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()

    return {"eval accuracy": acc.item()}


def run_generation(cfg, batch, model, tokenizer):
    input_ids = batch["input_ids"]
    input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    split_symbol = " [/INST]" if cfg.model_family == 'llama2-7b' else 'Answer: '
    ground_truth = [s.split(split_symbol)[1] for s in input_strings]
    input_strings = [s.split(split_symbol)[0] for s in input_strings]
    #add ["/INST "] to the end of each string
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


from datasets import load_dataset
import os
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


# 创建相关的数据

# dataset = load_dataset("locuslab/TOFU","forget10")["train"]
# answers_file = 'data/tofu/answers_forget10.json'
# answers_file = os.path.expanduser(answers_file)
# if os.path.exists(answers_file):
#     os.remove(answers_file)
# os.makedirs(os.path.dirname(answers_file), exist_ok=True)
# with open(answers_file, 'w') as file:
#     for ele in dataset:
#         question = ele["question"]
#         answer = ele["answer"]
#         json.dump(ele, file)
#         file.write('\n')
#         qingli = 3


# 读取响应的模型
# def test_tofu_model(model="llama"):
#     if model == "llama":
#         model_name = "locuslab/tofu_ft_llama2-7b"
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#     else:
#         model_name = "locuslab/tofu_ft_phi-1.5"
#         tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

#     model = AutoModelForCausalLM.from_pretrained(model_name)

#     inputs = tokenizer.encode(
#         "Can you tell us about the type of books that Jaime Vasquez writes?",
#         return_tensors='pt')
#     outputs = model.generate(inputs, max_length=50)
#     print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def test_tofu_model(model="llama"):
    revision = "checkpoint-12"  #30
    if model == "llama":
        model_name = "locuslab/tofu_ft_llama2-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model_name = "locuslab/tofu_ft_phi-1.5"
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
        # model_name = "locuslab/phi_idk_1e-05_forget05"
        # model_name = "locuslab/phi_grad_ascent_1e-05_forget05"
        # tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)

    # text = "What is the full name of the geology author born in Karachi, Pakistan on 06/30/1975?"
    # text = "1975 年 6 月 30 日出生于Karachi, Pakistan的地质学作家的全名是什么？"
    text = "In which genre does Hina Ameen primarily write?"
    # text = "What is the main genre of literature that Hina Ameen focuses on in her writings?"
    # text = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. Please give me the answer of the following question: What is the main genre of literature that Hina Ameen focuses on in her writings?"
    # text = "Hina Ameen 主要创作哪种类型的作品？"
    # text = "In welchem ​​Genre schreibt Hina Ameen hauptsächlich?"
    inputs = tokenizer.encode(text,return_tensors='pt')
    outputs = model.generate(inputs, max_length=150)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))



if __name__ == "__main__":
    test_tofu_model(model="phi")
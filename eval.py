

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,set_seed,DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training
from datasets import load_dataset
import transformers
import time
import sys
import os
import logging
import numpy as np
from argparse import ArgumentParser
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
logger = logging.getLogger(__name__)


parser = ArgumentParser(description='Process some integers.')

parser.add_argument('--model_name_or_path', default="wenge-research/yayi-7b", type=str, help='model path')
parser.add_argument('--prompt_column', default="content", type=str, help='get the column names for input/target.')
parser.add_argument('--response_column', default='summary', type=str, help='answer in the data')
parser.add_argument('--history_column', default=None, type=str, help='history in the data')
parser.add_argument('--ignore_pad_token_for_loss', default=True, type=bool, help='add pad_token need to calculate loss')
parser.add_argument('--max_target_length', default=64, type=int, help='output maximum length')
parser.add_argument('--max_source_length', default=128, type=int, help='input maximum length')
parser.add_argument('--eval_file', default='AdvertiseGen/test.json', type=str, help='dev dataset path')
parser.add_argument('--seed', default=42, type=int, help='seed')
parser.add_argument('--result_path',default='outputs/result.csv', type=str, help='result save path')
parser.add_argument('--quan', default='nf4', type=str, help='nf4 or fp4')
parser.add_argument('--double_quant', default=True, type=bool, help='double quant')
parser.add_argument('--peft_model_id', default="best_model", type=str, help='lora model path')
parser.add_argument('--end_token', default=25000, type=int, help='default end_token')
# Load dataset
def load(data_file,is_train=False):
    data_files = {}
    if data_file is not None:
        if is_train:
            data_files["train"] = data_file
            extension = data_file.split(".")[-1]
        else:
            data_files["eval"] = data_file
            extension = data_file.split(".")[-1]

    raw_datasets = load_dataset(
        extension,
        data_files=data_files
    )
    return raw_datasets



def print_dataset_example(example,tokenizer):
    print("input_ids",example["input_ids"])
    print("inputs", tokenizer.decode(example["input_ids"]),np.array(example["input_ids"]).shape)
    print("label_ids", example["labels"])
    print("labels", tokenizer.decode(example["labels"]),np.array(example["labels"]).shape)


def main():
    args = parser.parse_args()

    #quan config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=args.double_quant,
        bnb_4bit_quant_type=args.quan,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load pretrained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, quantization_config=bnb_config, device_map='auto', trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.peft_model_id)
    # Set seed before initializing model.
    set_seed(args.seed)

    def preprocess_function_eval(examples):
        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[args.prompt_column])):
            if examples[args.prompt_column][i] and examples[args.response_column][i]:
                query = examples[args.prompt_column][i]
                if args.history_column is None or len(examples[args.history_column][i]) == 0:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[args.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                
                input_id = tokenizer.encode(text=prompt, add_special_tokens=False)+[args.end_token]
                target = tokenizer.encode(text=examples[args.response_column][i], add_special_tokens=False)
                model_inputs["input_ids"].append(input_id)
                model_inputs["labels"].append(target)

        if args.ignore_pad_token_for_loss:
            model_inputs["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else 0) for l in label] for label in model_inputs["input_ids"]
            ]
        return model_inputs

    #load data
    raw_datasets = load(args.eval_file)

    column_names = raw_datasets["eval"].column_names
    eval_dataset = raw_datasets["eval"]

    eval_dataset = eval_dataset.map(
        preprocess_function_eval,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dev dataset",
    )
    print_dataset_example(eval_dataset[0],tokenizer)

    

    start_time = time.time()

    

    # Data collator
    label_pad_token_id = 0 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=True
    )
    dataload = DataLoader(
        eval_dataset,
        batch_size=2,
        collate_fn=data_collator,
    )

    # eval
    model.eval()
    with torch.no_grad():
        decoded_outputs = []
        decoded_labels = []
        end_flag = tokenizer.decode(args.end_token)
        print(end_flag)
        for id,datas in tqdm(enumerate(dataload)):
            
            input_id = datas['input_ids'].to("cuda")
            label = datas['labels'].to("cuda")
            outputs = model.generate(input_ids=input_id, max_new_tokens=args.max_target_length)
                
            decoded_output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
            decoded_label = tokenizer.batch_decode(label.detach().cpu().numpy(), skip_special_tokens=True)

            # print(decoded_output,decoded_label)
            for index in range(len(decoded_output)):
                decoded_output[index] = decoded_output[index].split(end_flag)[1]
                if decoded_output[index] == "":
                    decoded_output[index] = "当前的问题，yayi不知道回答。"
                decoded_outputs.append(decoded_output[index])
                decoded_labels.append(decoded_label[index])
        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        for pred, label in zip(decoded_outputs, decoded_labels):
            
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
            result = scores[0]
            
            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        print('score_dict',score_dict)


        with open(args.result_path, 'w', newline='') as csvfile:
            writer  = csv.writer(csvfile)
            for index in range(len(decoded_outputs)):
                writer.writerow(["predict: "+decoded_outputs[index]+" label: "+decoded_labels[index]])

        print("保存结果成功")


    end_time = time.time()
    print("耗时: {:.2f}秒".format(end_time - start_time))


if __name__ == "__main__":
    main()
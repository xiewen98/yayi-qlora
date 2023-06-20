

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

logger = logging.getLogger(__name__)


parser = ArgumentParser(description='Process some integers.')

parser.add_argument('--model_name_or_path', default="wenge-research/yayi-7b", type=str, help='model path')
parser.add_argument('--prompt_column', default="content", type=str, help='get the column names for input/target.')
parser.add_argument('--response_column', default='summary', type=str, help='answer in the data')
parser.add_argument('--history_column', default=None, type=str, help='history in the data')
parser.add_argument('--ignore_pad_token_for_loss', default=True, type=bool, help='add pad_token need to calculate loss')
parser.add_argument('--max_target_length', default=128, type=int, help='output maximum length')
parser.add_argument('--max_source_length', default=64, type=int, help='input maximum length')
parser.add_argument('--train_file', default='AdvertiseGen/train.json', type=str, help='train dataset path')
parser.add_argument('--seed', default=42, type=int, help='seed')
parser.add_argument('--quan', default='nf4', type=str, help='nf4 or fp4')
parser.add_argument('--double_quant', default=True, type=bool, help='double quant')
parser.add_argument('--lora_r', default=16, type=int, help='low rank')
parser.add_argument('--lora_alpha', default=32, type=int, help='normalized lora rank ')
parser.add_argument('--epoch', default=200, type=int, help='train epoch')
parser.add_argument('--batch_size', default=2, type=int, help='batch size')
parser.add_argument('--gradient_steps', default=4, type=int, help='end_batch is batch_size*gradient_steps')
parser.add_argument('--out_dir', default='outputs', type=str, help='lora save path')
parser.add_argument('--lr', default=2e-4, type=float, help='learn rate')

# Load dataset
def load(train_file):
    data_files = {}
    if train_file is not None:
        data_files["train"] = train_file
        extension = train_file.split(".")[-1]

    raw_datasets = load_dataset(
        extension,
        data_files=data_files
    )

    return raw_datasets

def print_dataset_example(example,tokenizer):
    print("input_ids",example["input_ids"])
    print("inputs", tokenizer.decode(example["input_ids"]))
    print("label_ids", example["labels"])
    print("labels", tokenizer.decode(example["labels"]))


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



def main():
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=args.double_quant,
        bnb_4bit_quant_type=args.quan,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    config = LoraConfig(
        r=args.lora_r, 
        lora_alpha=args.lora_alpha, 
        target_modules=["query_key_value"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Set seed before initializing model.
    set_seed(args.seed)

    def preprocess_function_train(examples):
        max_seq_length = args.max_source_length + args.max_target_length

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[args.prompt_column])):
            if examples[args.prompt_column][i] and examples[args.response_column][i]:
                query, answer = examples[args.prompt_column][i], examples[args.response_column][i]

                if args.history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[args.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False)


                if len(a_ids) > args.max_source_length - 1:
                    a_ids = a_ids[: args.max_source_length - 1]

                if len(b_ids) > args.max_target_length - 2:
                    b_ids = b_ids[: args.max_target_length - 2]
                input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

                context_length = len(a_ids)
                mask_position = context_length - 1
                labels = [0] * context_length + input_ids[mask_position+1:]
                
                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
                if args.ignore_pad_token_for_loss:
                    labels = [(l if l != tokenizer.pad_token_id else 0) for l in labels]
                # print("labels:",labels)
                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs

    #load data
    raw_datasets = load(args.train_file)
    # Preprocessing the datasets.
    column_names = raw_datasets["train"].column_names

    train_dataset = raw_datasets["train"]
    # with main_process_first(desc="train dataset map pre-processing"):
    train_dataset = train_dataset.map(
        preprocess_function_train,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on train dataset",
    )
    print_dataset_example(train_dataset[0],tokenizer)

    # Data collator
    label_pad_token_id = 0 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, quantization_config=bnb_config, device_map='auto')
    # Training
    start_time = time.time()
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_steps,
            warmup_steps=2,
            max_steps=args.epoch,
            learning_rate=args.lr,
            fp16=True,
            output_dir=args.out_dir,
            optim="paged_adamw_8bit"
        ),
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=None,
            padding=False
        )
    )
    train_result = trainer.train()

    model.save_pretrained("best_model")
    end_time = time.time()
    print("耗时: {:.2f}秒".format(end_time - start_time))


if __name__ == "__main__":
    main()
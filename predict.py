import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import os
import numpy as np


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

peft_model_id = "best_model"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,quantization_config=bnb_config, device_map='auto')
model = PeftModel.from_pretrained(model, peft_model_id)



tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model.eval()
with torch.no_grad():
    prompt = "类型#上衣*风格#嘻哈*图案#卡通*图案#印花*图案#撞色*衣样式#卫衣*衣款式#连帽"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=100)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])



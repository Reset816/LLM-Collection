from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

checkpoint = "bigcode/starcoder"

# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForCausalLM.from_pretrained(checkpoint,
#                                             trust_remote_code=True,
#                                             device_map="auto",)

# inputs = tokenizer.encode("write a python function of hello world", return_tensors="pt")
# inputs = inputs.to('cuda')
# outputs = model.generate(inputs,max_new_tokens=50)
# print(tokenizer.decode(outputs[0]))

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False, legacy=False)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<fim_pad>")
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
tokenizer.pad_token = "<fim_pad>"
tokenizer.eos_token = "<|endoftext|>"
# try 4bit loading if cuda memory not enough
model = AutoModelForCausalLM.from_pretrained(checkpoint,
                                             trust_remote_code=True,
                                             load_in_4bit=False,
                                             device_map="auto",
                                             torch_dtype=torch.bfloat16)
model.eval()

HUMAN_ROLE_START_TAG = "<|role_start|>human<|role_end|>"
BOT_ROLE_START_TAG = "<|role_start|>bot<|role_end|>"

text = f"{HUMAN_ROLE_START_TAG}write a python function of quick sort.{BOT_ROLE_START_TAG}" 
inputs = tokenizer(text, return_tensors='pt', padding=True, add_special_tokens=False).to("cuda")
outputs = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,
        top_p=0.95,
        temperature=0.1,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
gen_text = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(gen_text[0])

from llm import *


class SecGPT(LLM):
    def __init__(self):
        self.load_model()

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        from peft import PeftModel

        self.tokenizer = AutoTokenizer.from_pretrained(
            "w8ay/secgpt", trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "w8ay/secgpt",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    @cost_time
    def run_model(self, input_text):
        def reformat_sft(instruction, input):
            if input:
                prefix = (
                    "Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n"
                    f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
                )
            else:
                prefix = (
                    "Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\n"
                    f"### Instruction:\n{instruction}\n\n### Response:"
                )
            return prefix

        query = self.system_prompt + reformat_sft(input_text, "")

        generation_kwargs = {
            "top_p": 0.7,
            "temperature": 0.3,
            "max_new_tokens": self.max_new_tokens,
            "max_length": self.max_length,
            "max_time": self.max_time,
            "do_sample": True,
            "repetition_penalty": 1.1,
        }

        inputs = self.tokenizer.encode(query, return_tensors="pt", truncation=True)
        inputs = inputs.cuda()
        generate = self.model.generate(input_ids=inputs, **generation_kwargs)
        return self.tokenizer.decode(generate[0])

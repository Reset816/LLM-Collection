from llm import *


class CodeLlama_70b(LLM):
    def __init__(self):
        self.load_model()

    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.model = AutoModelForCausalLM.from_pretrained(
            "codellama/CodeLlama-70b-Instruct-hf",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "codellama/CodeLlama-70b-Instruct-hf"
        )

    @cost_time
    def run_model(self, input_text):
        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text},
        ]
        inputs = self.tokenizer.apply_chat_template(chat, return_tensors="pt")
        output = self.model.generate(
            input_ids=inputs,
            max_new_tokens=self.max_new_tokens,
            max_length=self.max_length,
            max_time=self.max_time,
        )
        output = output[0].to("cpu")
        return self.tokenizer.decode(output)

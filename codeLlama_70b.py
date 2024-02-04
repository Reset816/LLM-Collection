from llm import *


class CodeLlama_70b(LLM):
    def __init__(self):
        self.load_model()

    def load_model(self):
        import torch
        from modelscope import AutoTokenizer, Model

        self.model = Model.from_pretrained(
            "AI-ModelScope/CodeLlama-70b-Instruct-hf",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "AI-ModelScope/CodeLlama-70b-Instruct-hf"
        )

    @cost_time
    def run_model(self, input_text):
        inputs = {
            "system": self.system_prompt,
            "text": input_text,
            "do_sample": True,
            "top_k": 10,
            "temperature": 0.1,
            "top_p": 0.95,
            "num_return_sequences": 1,
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": self.max_new_tokens,
            "max_length": self.max_length,
            "max_time": self.max_time,
        }

        output = self.model.chat(inputs, self.tokenizer)
        return output["response"]

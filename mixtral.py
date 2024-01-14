from llm import *


class Mixtral(LLM):
    def __init__(self):
        self.load_model()

    def load_model(self):
        import torch
        from modelscope import AutoTokenizer, AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(
            "AI-ModelScope/Mixtral-8x7B-Instruct-v0.1",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "AI-ModelScope/Mixtral-8x7B-Instruct-v0.1"
        )

    @cost_time
    def run_model(self, input_text):
        # reference: https://web.archive.org/web/20231030013339/https://docs.mistral.ai/usage/guardrailing/
        inputs = self.tokenizer(
            "[INST] " + self.system_prompt + "\n" + input_text + " [/INST]",
            return_tensors="pt",
        ).to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            max_length=self.max_length,
            max_time=self.max_time,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

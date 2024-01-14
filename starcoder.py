from llm import *


class StarCoder(LLM):
    def __init__(self):
        self.load_model()

    def load_model(self):
        import torch
        from modelscope import AutoTokenizer, AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(
            "codefuse-ai/CodeFuse-StarCoder-15B",
            trust_remote_code=True,
            load_in_4bit=False,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "codefuse-ai/CodeFuse-StarCoder-15B",
            trust_remote_code=True,
            use_fast=False,
            legacy=False,
        )

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("<fim_pad>")
        self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|endoftext|>"
        )
        self.tokenizer.pad_token = "<fim_pad>"
        self.tokenizer.eos_token = "<|endoftext|>"

    @cost_time
    def run_model(self, input_text):
        HUMAN_ROLE_START_TAG = "<|role_start|>human<|role_end|>"
        BOT_ROLE_START_TAG = "<|role_start|>bot<|role_end|>"

        text = f"{HUMAN_ROLE_START_TAG}{input_text}{BOT_ROLE_START_TAG}"
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, add_special_tokens=False
        ).to("cuda")
        outputs = self.model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            max_length=self.max_length,
            max_time=self.max_time,
            top_p=0.95,
            temperature=0.1,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_text = self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return gen_text[0]

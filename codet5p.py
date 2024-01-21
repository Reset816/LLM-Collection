from llm import *


class CodeT5p(LLM):
    def __init__(self):
        self.load_model()

    def load_model(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        checkpoint = "Salesforce/instructcodet5p-16b"

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint, trust_remote_code=True, device_map="auto"
        )

    @cost_time
    def run_model(self, user_prompt, input_text):
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        )
        formated_prompt = prompt.format_map(
            {"instruction": user_prompt, "input": input_text}
        )
        self.encoding = self.tokenizer(formated_prompt, return_tensors="pt").to("cuda")
        self.encoding["decoder_input_ids"] = self.encoding["input_ids"]
        outputs = self.model.generate(
            **self.encoding,
            max_new_tokens=self.max_new_tokens,
            max_length=self.max_length,
            max_time=self.max_time,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

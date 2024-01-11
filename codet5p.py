class CodeT5p:
    def __init__(self):
        self.load_model()

    def load_model(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        checkpoint = "Salesforce/codet5p-16b"

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint, trust_remote_code=True, device_map="auto"
        )

    def run_model(self, input_text):
        self.encoding = self.tokenizer(input_text, return_tensors="pt").to("cuda")
        self.encoding["decoder_input_ids"] = self.encoding["input_ids"]
        outputs = self.model.generate(**self.encoding, max_length=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

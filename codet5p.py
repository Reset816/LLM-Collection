from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

checkpoint = "Salesforce/codet5p-16b"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                              trust_remote_code=True,
                                              device_map="auto")

encoding = tokenizer("write a python function of quick sort.", return_tensors="pt")
encoding['decoder_input_ids'] = encoding['input_ids'].clone()
outputs = model.generate(**encoding, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

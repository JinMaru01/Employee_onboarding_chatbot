from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load pre-trained T5 model fine-tuned for paraphrasing (or base T5)
tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5_paraphraser")

def paraphrase_t5(text, num_return_sequences=5, num_beams=5):
    input_text = f"paraphrase: {text} </s>"
    encoding = tokenizer.encode_plus(input_text, return_tensors="pt")
    
    outputs = model.generate(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        max_length=64,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams
    )

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
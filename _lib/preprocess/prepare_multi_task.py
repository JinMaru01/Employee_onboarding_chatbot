import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from _lib.models.Multi_Tasks import MultitaskDataset

def prepare_multitask_data(json_data, tokenizer_name="distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    intent_encoder = LabelEncoder()
    tag_encoder = LabelEncoder()

    texts = [item["text"] for item in json_data]
    intents = [item["intent"] for item in json_data]
    all_tags = [tag for item in json_data for tag in item["tags"]]

    # Fit encoders
    intent_encoder.fit(intents)
    tag_encoder.fit(all_tags)

    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", is_split_into_words=False)
    ner_labels = []

    for i, item in enumerate(json_data):
        word_ids = encodings.word_ids(batch_index=i)
        tag_ids = []
        tag_seq = item["tags"]

        word_idx_map = []
        current_word = -1
        for idx in word_ids:
            if idx is None:
                tag_ids.append(-100)
            elif idx != current_word:
                current_word = idx
                if current_word < len(tag_seq):
                    tag_ids.append(tag_encoder.transform([tag_seq[current_word]])[0])
                else:
                    tag_ids.append(-100)
            else:
                if current_word < len(tag_seq):
                    tag_ids.append(tag_encoder.transform([tag_seq[current_word]])[0])
                else:
                    tag_ids.append(-100)


        ner_labels.append(tag_ids)

    dataset = MultitaskDataset(
        encodings=encodings.data,
        intent_labels=torch.tensor(intent_encoder.transform(intents)),
        ner_labels=torch.tensor(ner_labels)
    )

    return dataset, intent_encoder, tag_encoder
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification

from extend_path import sys
from _lib.database.redis_conn import RedisConn
from _lib.models.Entity_Recognition import NamedEntityRecognizer

# Initialize Redis connection
redis_con = RedisConn()

# Load train and test datasets from Redis
train_dataset = redis_con.label_encoder_load("ner_train_dataset")
test_dataset = redis_con.label_encoder_load("ner_test_dataset")

label2id = redis_con.label_encoder_load("ner_label2id")
id2label = redis_con.label_encoder_load("ner_id2label")

tokenizer = redis_con.label_encoder_load("tokenizer")
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

model_checkpoint = "distilbert-base-uncased"
batch_size = 32
num_epochs = 5
num_labels = len(id2label)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5
)

extractor = NamedEntityRecognizer(model, tokenizer, id2label, device)

model = extractor.train(train_loader, optimizer, 10)

torch.save(model, "./artifact/model/entity_extractor.pth")
from _lib.models.Intent_Classification import IntentClassifier
from _lib.models.Entity_Recognition import NamedEntityRecognizer

class Loader():
    def __init__(self, con):

        # Load Classifier Model
        self.classifier = con.classifier_load("intent_classifier_v2")
        self.tokenizer = con.label_encoder_load("tokenizer")
        self.label_encoder = con.label_encoder_load("label-encoder")

        # Load Extractor Model
        self.extractor =  con.extractor_load("extractor")
        self.id2label = con.label_encoder_load("ner_id2label")

        # Initial Model
        self.classifier = IntentClassifier(self.classifier, self.tokenizer, self.label_encoder)
        self.extractor = NamedEntityRecognizer(self.extractor, self.tokenizer, self.id2label)
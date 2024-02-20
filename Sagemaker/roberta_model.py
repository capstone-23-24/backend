import torch
import torch.nn as nn
from transformers import RobertaForTokenClassification

class MyModel(nn.Module):
    def __init__(self, num_labels):
        super(MyModel, self).__init__()
        self.num_labels = num_labels
        self.label_map = {
            'O': 0,          # Outside of any named entity
            'Person': 1,     # Beginning of a name
            'Location': 2,      # Beginning of a location
            '-100': -100     # Special token used to ignore subtokens in loss calculation
        }
        self.roberta = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=len(self.label_map))
        self.roberta.config.id2label = { v:k for k, v in self.label_map.items() }
        self.roberta.config.label2id = self.label_map
        self.config = self.roberta.config


    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        if labels is not None:
            loss = outputs.loss
            return {"loss": loss, "logits": outputs.logits}
        else:
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=-1).tolist()
            return {"logits": logits, "predicted_labels": predicted_labels}
        
    def save_model(self, output_dir):
        self.roberta.save_pretrained(output_dir)

import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification

class MyModel(nn.Module):
    def __init__(self, num_labels):
        super(MyModel, self).__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        if labels is not None:
            # Compute the loss if labels are provided
            loss = outputs.loss
            return {"loss": loss}
        else:
            # During inference, return the logits
            logits = outputs.logits
            return {"logits": logits}
        
    def save_model(self, output_dir):
        self.roberta.save_pretrained(output_dir)

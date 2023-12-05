import torch.nn as nn
from transformers import RobertaForSequenceClassification

class MyModel(nn.Module):
    def __init__(self, num_labels):
        super(MyModel, self).__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)

    def forward(self, input_ids_part1, attention_mask_part1, input_ids_part2, attention_mask_part2):
        outputs = self.roberta(input_ids=input_ids_part1, attention_mask=attention_mask_part1)
        logits_part1 = outputs.logits

        outputs = self.roberta(input_ids=input_ids_part2, attention_mask=attention_mask_part2)
        logits_part2 = outputs.logits

        # Combine or process the logits from different parts as needed
        # For example, concatenate them or apply some operation
        combined_logits = torch.cat([logits_part1, logits_part2], dim=1)

        return combined_logits


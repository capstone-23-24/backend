import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, case, labels, tokenizer, max_length=512):
        self.case = case
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.case)

    def __getitem__(self, idx):
        case_text = str(self.case[idx])

        try:
            labels = self.labels[idx]
        except KeyError:
            labels = 0

        # Tokenize the case text
        encoding = self.tokenizer.encode_plus(
            case_text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
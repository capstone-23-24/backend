import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast

class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer  # This should be an instance of RobertaTokenizerFast
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        # Tokenize the text using RobertaTokenizerFast
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # Initialize label_sequence with -100 to ignore in loss calculation
        label_sequence = torch.ones(self.max_length, dtype=torch.long) * -100
        
        # Copy labels to the label_sequence according to the tokenization mapping
        label_sequence[:len(labels)] = torch.tensor(labels, dtype=torch.long)[:self.max_length]

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_sequence
        }

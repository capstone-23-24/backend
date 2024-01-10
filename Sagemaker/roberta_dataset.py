from torch.utils.data import Dataset
        
class MyDataset(Dataset):
    def __init__(self, case, labels):
        self.case = case
        self.labels = labels

    def __len__(self):
        return len(self.case)

    def __getitem__(self, idx):
        return {
            'case': self.case[idx],
            'labels': self.labels[idx]
        }


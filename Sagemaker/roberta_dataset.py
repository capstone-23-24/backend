from torch.utils.data import Dataset
        
class MyDataset(Dataset):
    def __init__(self, case_part1, case_part2, labels):
        self.case_part1 = case_part1
        self.case_part2 = case_part2
        self.labels = labels

    def __len__(self):
        return len(self.case_part1)

    def __getitem__(self, idx):
        return {
            'case_part1': self.case_part1[idx],
            'case_part2': self.case_part2[idx],
            'labels': self.labels[idx]
        }


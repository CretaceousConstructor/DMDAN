import torch.utils.data


class BCI_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.len = len(data)
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.Data[index], self.Label[index]

    def __len__(self):
        return self.len



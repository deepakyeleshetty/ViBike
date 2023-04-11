from torch.utils.data import Dataset, DataLoader


class IMUDataset(Dataset):
    def __init__(self, x, y, transforms=None):
        self.x = x
        self.y = y
        self.transforms = transforms

    def __getitem__(self, index):
        inp = self.x[index]
        label = self.y[index]
        return inp, label

    def __len__(self):
        return self.y.shape[0]
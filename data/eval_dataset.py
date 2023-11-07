from torch.utils.data import Dataset


class RealEvaluationDataset(Dataset):
    def __init__(self, public_x, public_y, hidden_x, hidden_y):
        super().__init__()

        self.xc, self.yc = public_x, public_y
        self.xt, self.yt = hidden_x, hidden_y
    
    def __len__(self):
        return self.xc.shape[0]

    def __getitem__(self, idx):
        xc, yc = self.xc[idx], self.yc[idx]
        xt, yt = self.xt[idx], self.yt[idx]

        return xc, yc, xt, yt
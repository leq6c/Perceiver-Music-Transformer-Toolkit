from torch.utils.data import DataLoader, Dataset

# helpers
def cycle(loader):
    while True:
        for data in loader:
            yield data

# Dataloader
class MusicDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):

        # random sampling
        # idx = secrets.randbelow((self.data.size(0) // (self.seq_len))-1) * (self.seq_len)

        # consequtive sampling seems to be better at 64k seq_len
        idx = index * self.seq_len

        full_seq = self.data[idx: idx + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return (self.data.size(0) // self.seq_len)-1

def prepare_train_data(train_data, SEQ_LEN, BATCH_SIZE):
    train_dataset = MusicDataset(train_data, SEQ_LEN)
    val_dataset   = MusicDataset(train_data, SEQ_LEN)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))
    return train_loader, val_loader

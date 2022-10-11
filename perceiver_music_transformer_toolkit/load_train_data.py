import torch
import tqdm
import pickle
import os

def load_train_data(directory):
    filez = list()
    for (dirpath, dirnames, filenames) in os.walk(directory):
        filez += [os.path.join(dirpath, file) for file in filenames]
    print('=' * 70)

    filez.sort()

    print('Loading training data... Please wait...')

    train_data = torch.Tensor()

    for f in tqdm.tqdm(filez):
        train_data = torch.cat((train_data, torch.Tensor(pickle.load(open(f, 'rb')))))
        print('Loaded file:', f)

    print('Done!')
    
    return train_data
import pickle
import os
import random
import secrets
import tqdm
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn import metrics
from tegridy_tools import TMIDIX

def train(model, optim, train_loader, val_loader, NUM_BATCHES, GRADIENT_ACCUMULATE_EVERY, VALIDATE_EVERY, GENERATE_EVERY, GENERATE_LENGTH, SAVE_EVERY, checkpoint_dir, enable_plt=False):
    # Train the model
    train_losses = []
    val_losses = []

    train_accs = []
    val_accs = []

    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='Training'):
        model.train()

        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            loss, acc = model(next(train_loader))
            loss.backward()

        print(f'Training loss: {loss.item()}')
        print(f'Training acc: {acc.item()}')

        train_losses.append(loss.item())
        train_accs.append(acc.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_acc = model(next(val_loader))
                print(f'Validation loss: {val_loss.item()}')
                print(f'Validation acc: {val_acc.item()}')
                val_losses.append(val_loss.item())
                val_accs.append(val_acc.item())

                if enable_plt:
                    print('Plotting training loss graph...')

                    tr_loss_list = train_losses
                    plt.plot([i for i in range(len(tr_loss_list))] ,tr_loss_list, 'b')
                    plt.show()
                    # plt.savefig('/notebooks/training_loss_graph.png')
                    plt.close()
                    print('Done!')

                    print('Plotting training acc graph...')

                    tr_loss_list = train_accs
                    plt.plot([i for i in range(len(tr_loss_list))] ,tr_loss_list, 'b')
                    plt.show()
                    # plt.savefig('/notebooks/training_acc_graph.png')
                    plt.close()
                    print('Done!')

                    print('Plotting validation loss graph...')
                    tr_loss_list = val_losses
                    plt.plot([i for i in range(len(tr_loss_list))] ,tr_loss_list, 'b')
                    plt.show()
                    # plt.savefig('/notebooks/validation_loss_graph.png')
                    plt.close()
                    print('Done!')

                    print('Plotting validation acc graph...')
                    tr_loss_list = val_accs
                    plt.plot([i for i in range(len(tr_loss_list))] ,tr_loss_list, 'b')
                    plt.show()
                    # plt.savefig('/notebooks/validation_accs_graph.png')
                    plt.close()
                    print('Done!')

        """
        if i % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            print(inp)
            sample = model.generate(inp[None, ...], GENERATE_LENGTH)
            print(sample)
        """

        if i % SAVE_EVERY == 0:
            print('Saving model progress. Please wait...')
            print('model_checkpoint_' + str(i) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss.pth')
            fname = os.path.join(checkpoint_dir, 'model_checkpoint_'  + str(i) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss.pth')
            torch.save(model.state_dict(), fname)
            print('Done!')

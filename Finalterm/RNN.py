import torch
import torch.nn as nn
import torch.optim as optim
import DataLoad as DL
from train import RNN_train_one_epoch, RNN_evaluate, Submit, Early_stop
from Model import RNN
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import copy
import time
import datetime

device = 'cuda' if torch. cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)

import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('AI_Final_RNN ', add_help=False)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100,type=int)
    parser.add_argument('--hiddensize', default=128,type=int)
    parser.add_argument('--num_layers', default=1,type=int)
    parser.add_argument('--dropout', default=0,type=float)
    parser.add_argument('--window', default=30,type=int)
    parser.add_argument('--wandbflag', default=False,type=bool)
    return parser

parser = argparse.ArgumentParser('AI_Final_RNN', parents=[get_args_parser()])
args = parser.parse_args(args=[])

if args.wandbflag:
    import wandb
    print("Wandb On")
    wandb.login()
    wandb.init(project='23_2 AI Final project',
            config=args,
            reinit=True)

    config = wandb.config
else:
    config = args

#############################################################################################
df = np.loadtxt('AI_Lec_23_final_stocks.csv', delimiter='\t')

scaler = MinMaxScaler()
df = scaler.fit_transform(df)

leng = df.shape[0]
columns = df.shape[1]
window = config.window

total = np.empty((20,1))

start = time.time()

for column in range(columns):
    print(column+1)
    data = df[:,column]
    
    train_len = int(0.95*leng)
    train_loader = DL.loader(DL.StockDataset(data[:train_len], window), batch_size = config.batch_size)

    test_list = data[train_len-window:]
    test_list = torch.tensor(test_list, dtype=torch.float32).unsqueeze(0)

    input_size = 1
    hidden_size = config.hiddensize
    output_size = 1
    num_layers = config.num_layers
    model = RNN(input_size, hidden_size, output_size, num_layers, config.dropout)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    model=model.to(device)
    epochs = config.epochs

    train_loss_values = []
    test_loss_values = []
    best_loss_values = []
    patience_limit=3
    patience_check=0
    best_model = copy.deepcopy(model)
    best_loss = 10**3

    for epoch in range(epochs):
        train_loss = RNN_train_one_epoch(model, train_loader, optimizer, criterion)
        test_loss = RNN_evaluate(model, test_list, criterion, window)
        
        train_loss_values.append(train_loss)
        test_loss_values.append(test_loss)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Test Loss: {test_loss}')
        
        if config.wandbflag:
            wandb.log({"train_loss": train_loss, "test_loss": test_loss})
        
        best_model, best_loss, patience_check = Early_stop(
            model, best_model, best_loss, test_loss, patience_check
        )
        if patience_check >= patience_limit:
            break
        
    print(f'Best loss: {best_loss}')
    best_loss_values.append(best_loss)

    if config.wandbflag:
        wandb.log({"Best Loss": best_loss})

    partial_submit = Submit(best_model, test_list, window)
    total = np.concatenate((total, partial_submit.reshape(-1,1)), axis=1)

best_loss_values = np.mean(best_loss_values)

if config.wandbflag:
    wandb.log({"Total Loss": best_loss_values})

total = np.delete(total,0,1)
total = scaler.inverse_transform(total)
final = time.time()-start
timelist = str(datetime.timedelta(seconds=final)).split(".")
print(f'Training Time : {timelist[0]}')

# plot
import matplotlib.pyplot as plt
df = scaler.inverse_transform(df)
graph = np.concatenate((df, total), axis=0)
print(graph.shape)

np.savetxt('predict_submit.csv', total, delimiter=",")
# plt.plot(graph[-50:, 0:5])
# plt.show()

# plt.plot(range(len(train_loss_values)), train_loss_values, label='Train_Loss')
# plt.plot(range(len(val_loss_values)), val_loss_values, label='Val_Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.title("Training and Validation Loss Curves")
# plt.show()

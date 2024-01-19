import torch
import torch.nn as nn
import torch.optim as optim
import DataLoad as DL
from train import train_one_epoch, evaluate
from Model import FCNN

root_path = 'Data'
device = 'cuda' if torch. cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)

import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('AI_Midterm ', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=64, type=int )
    parser.add_argument('--epochs', default=300,type=int)
    parser.add_argument('--input_layer', default=128,type=int)
    parser.add_argument('--hidden_layer', default=32,type=int)
    parser.add_argument('--dropout', default=0.0,type=float)
    parser.add_argument('--wandbflag', default=False,type=bool)
    return parser

parser = argparse.ArgumentParser('AI_Midterm ', parents=[get_args_parser()])
args = parser.parse_args(args=[])

if args.wandbflag:
    import wandb
    print("Wandb On")
    wandb.login()
    wandb.init(project='23-2-AI-Midterm',
            config=args,
            reinit=True)

    config = wandb.config
else:
    config = args

img_size = 48*48*3
num_classes = 2

model = FCNN(img_size, num_classes, config.input_layer, config.hidden_layer, config.dropout)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.NAdam(model.parameters(), lr=config.lr)

train_dataset, validation_dataset, test_dataset = DL.split_datasets(DL.create_datasets(root_path))
train_dataset, validation_dataset, test_dataset = DL.augmentation_datasets(train_dataset, validation_dataset, test_dataset)
train_loader, val_loader, test_loader = DL.data_load(train_dataset, validation_dataset, test_dataset, config.batch_size)

print(len(train_dataset), len(validation_dataset), len(test_dataset))

model=model.to(device)
epochs = config.epochs

import copy

train_loss_values = []
val_loss_values = []

patience_limit=5
patience_check=0
best_loss = 10**3

for epoch in range(epochs):
    train_accuracy, train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_accuracy, val_loss = evaluate(model, val_loader, criterion)

    train_loss_values.append(train_loss)
    val_loss_values.append(val_loss)

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy:.2f}%')
    
    if config.wandbflag:
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_accuracy": train_accuracy, "val_accuracy": val_accuracy})
    
    # Save best model
    if epoch == 0:
        best_model = copy.deepcopy(model)
        best_loss = val_loss

    if best_loss > val_loss:
        best_model = copy.deepcopy(model)
        best_loss = val_loss
        patience_check = 0
    
    # Early terminate
    elif val_loss > best_loss:
        patience_check += 1
    
        if patience_check >= patience_limit:
            print("Early Terminate")
            break

    if config.wandbflag:
        wandb.log({"best_loss": best_loss})
    
test_accuracy, test_loss = evaluate(best_model, test_loader, criterion)
print(f'Best loss: {best_loss}, Test loss: {test_loss}, Test accuracy: {test_accuracy:.2f}%')

if config.wandbflag:
    wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})

# plot
import matplotlib.pyplot as plt

plt.plot(range(len(train_loss_values)), train_loss_values, label='Train_Loss')
plt.plot(range(len(val_loss_values)), val_loss_values, label='Val_Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Training and Validation Loss Curves")
plt.show()
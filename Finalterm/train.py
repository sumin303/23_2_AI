import torch
from tqdm.auto import tqdm
import copy

device = 'cuda' if torch. cuda.is_available() else 'cpu'

def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total, running_loss / len(train_loader)

def evaluate(model, data_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total, val_loss / len(data_loader)

def RNN_train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(2))
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(inputs)
    
def RNN_evaluate(model, test_list, criterion, window):
    model.eval()
    test_loss = 0.0
    leng = test_list.size(1)-window

    with torch.no_grad():
        for i in range(leng):
            input_sequence = test_list[:,i:i+window]
            input_sequence, test_list = input_sequence.to(device), test_list.to(device)
            predict = model(input_sequence.unsqueeze(2))
            loss = criterion(predict, test_list[:,window+i].unsqueeze(1))
            test_loss += loss.item()
    
    return test_loss

def Submit(model, test_list, window):
    model.eval()

    with torch.no_grad():
        for _ in range(20):
            input_sequence = test_list[:,-window:]
            input_sequence, test_list = input_sequence.to(device), test_list.to(device)
            predict = model(input_sequence.unsqueeze(2))
            test_list = torch.cat((test_list, predict), dim=1)

    test_list = test_list.squeeze()
    test_list = test_list[-20:].cpu().detach().numpy()

    return test_list

def Early_stop(model, best_model, best_loss, test_loss, patience_check):

    if best_loss > test_loss:
        best_model = copy.deepcopy(model)
        best_loss = test_loss
        patience_check = 0
    
    elif test_loss > best_loss:
        patience_check += 1
          
    return best_model, best_loss, patience_check
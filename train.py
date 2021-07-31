import torch
import time
from tqdm import tqdm

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs):
   
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    model.to(device)

    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    best_acc = 0.0

    dataloader = train_dl
    val_loader = valid_dl
    
    print('Training')
    print('=' * 60)
    model.train()
    #start = time.time()

    for epoch in range(1,epochs+1):
        loss_temp = 0.0
        acc_temp = 0.0
        time.sleep(0.2)
        with tqdm(dataloader,unit='batch') as pbar:
            # iterate over data
            pbar.set_description('Epoch: {}/{}'.format(epoch,epochs))
            for data in pbar:
                x = data['image'].to(device)
                y = data['label'].to(device)
                y = torch.squeeze(y,1)
                
                # forward pass
                outputs = model(x)

                # compute loss
                loss = loss_fn(outputs, y.long())

                # zero the gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #compute the accuracy
                acc = acc_fn(outputs, y)

                acc_temp  += acc
                loss_temp += loss 

                pbar.set_postfix(loss=loss.item(), accuracy=100. * acc.item())
   
              
            running_loss = loss_temp / len(dataloader)
            running_acc = acc_temp / len(dataloader)
            pbar.set_postfix(loss=running_loss, accuracy=100. *running_acc)
            #print('Epoch {} Loss: {:.4f} Acc: {}'.format(epoch, running_loss, running_acc))

            print('Validation')
            print('=' * 60)

            val_loss = 0
            val_acc = 0

            for val_data in val_loader:
                x_val = val_data['image'].to(device)
                y_val = val_data['label'].to(device)
                y_val = torch.squeeze(y_val,1)

                # forward pass
                val_outputs = model(x_val)
                with torch.no_grad():
                    # compute loss
                    val_loss_temp = loss_fn(val_outputs, y_val.long())
                    val_acc_temp = acc_fn(val_outputs, y_val)

                    val_loss += val_loss_temp
                    val_acc += val_acc_temp

            running_val_loss = val_loss/len(val_loader)
            running_val_acc = val_acc/len(val_loader)

            print('Validation Loss: {}  Validation Acc: {}'.format(running_val_loss, running_val_acc))
            train_loss.append(running_loss) 
            train_acc.append(running_acc)

            valid_loss.append(running_val_loss)
            valid_acc.append(running_val_acc)
 
    train_loss_ = [train_loss[i].cpu().detach().numpy() for i in range(len(train_loss))]
    valid_loss_ = [valid_loss[i].cpu().detach().numpy() for i in range(len(valid_loss))]
    train_acc_ = [train_acc[i].cpu().detach().numpy() for i in range(len(train_acc))]
    valid_acc_ = [valid_acc[i].cpu().detach().numpy() for i in range(len(valid_acc))]
    
    return train_loss_, valid_loss_, train_acc_, valid_acc_    

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb).float().mean()

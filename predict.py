import torch
from utils import *

def calc_acc(test_loader,model,loss_fn,acc_fn):
    test_loss = 0
    test_acc = 0

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
        
    for test_data in test_loader:
        x_test = test_data['image'].to(device)
        y_test = test_data['label'].to(device)
        y_test = torch.squeeze(y_test,1)

        # forward pass
        y_hat = model(x_test)
        with torch.no_grad():
            # compute loss
            test_loss_temp = loss_fn(y_hat, y_test.long())
            test_acc_temp = acc_fn(y_hat, y_test)

            test_loss += test_loss_temp
            test_acc += test_acc_temp

    test_loss = test_loss/len(test_loader)
    test_acc = test_acc/len(test_loader)
    return test_loss, test_acc

def predict(model, image):
    y_hat = model(image.to(device))
    open_segmented_image(y_hat.cpu())

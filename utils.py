import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_raster, reshape_as_image
import torch
import os

def open_as_image(instance):
    if torch.is_tensor(instance):
        instance = instance.detach().numpy()

    if instance.shape[0] > 3:
        new_instance = np.dstack((instance[3],instance[2],instance[1]))
        return plt.imshow(new_instance)
    elif instance.shape[0] == 3:
        return plt.imshow(reshape_as_image(instance))
    else:
        return plt.imshow(instance[0])
            
def open_segmented_image(image):
    if torch.is_tensor(image):
        image = image.detach().numpy()

    if image.shape[0] == 3:
        image = image.argmax(0) #find the class using argmax across the channels
        return plt.imshow(image)
    else:
        return plt.imshow(image[0])
    
def plot_img(test_dataloader,model,n):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    test_data = next(iter(test_dataloader))
    
    model.to(device)
    y_hat = model(test_data['image'].to(device))
    
    if n>len(test_dataloader):
        n = len(test_dataloader)
        
    for j in range(n):
        plt.figure(figsize = (10, 8))
        plt.subplot(j+1,3,1)
        open_as_image(test_data['image'][j])
        plt.title('Pre-processed Original Image')
        plt.axis('off')

        plt.subplot(j+1,3,2)
        open_segmented_image(test_data['label'][j])
        plt.title('Original Mask')
        plt.axis('off')

        plt.subplot(j+1,3,3)
        open_segmented_image(y_hat[j].cpu())
        plt.title('Predicted Mask')
        plt.axis('off')
        
def plot_dataset(dataset,n):
    
    for j in range(n):
        plt.figure(figsize = (10, 8))
        plt.subplot(j+1,2,1)
        open_as_image(dataset[j]['image'])
        plt.title('Pre-processed Original Image')
        plt.axis('off')

        plt.subplot(j+1,2,2)
        open_segmented_image(dataset[j]['label'])
        plt.title('Original Mask')
        plt.axis('off')
        
        
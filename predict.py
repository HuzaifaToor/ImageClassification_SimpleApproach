#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 17:00:56 2022

@author: huzaifa
"""
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

#torch.load with map_location=torch.device('cpu')



def predict_image(image, model):
    device = torch.device('gpu')
    image_tensor = (image).float().transforms.ToTensor()
    image_tensor = image_tensor.unsqueeze_(0)
    #input = Variable(image_tensor)
    input = image_tensor.to(device)
    output = model(input)
    #print(output)
    index = output.data.cpu().numpy().argmax()
    #index = output.data.numpy().argmax().to(device)
    #print(index)
    return index



def get_random_images(num, data_dir):
    global classes
    data = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, 
                   sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels



def show_pred(data_path, model):


    data_dir = os.path.join(data_path + '/DataSet/test')
    classes = ('MitStift', 'OhneAdapter', 'OhneStift')



    to_pil = transforms.ToPILImage()
    images, labels = get_random_images(5, data_dir)
    fig=plt.figure(figsize=(20,20))
    for ii in range(len(images)):
        image = to_pil(images[ii])
        index = predict_image(image, model)
        sub = fig.add_subplot(1, len(images), ii+1)
        res = int(labels[ii]) == index
        sub.set_title(str(classes[index]) + ":" + str(res))
        plt.axis('off')
        plt.imshow(image)
    plt.show()
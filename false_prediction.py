#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 17:00:56 2022

@author: huzaifa
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

#torch.load with map_location=torch.device('cpu')
path = '/home/iuna/IUNA_AI/Classification/imagenet_myApproach'
device = torch.device('cpu')
model = torch.load(path + '/output/models/model_202209192220.pth', map_location=torch.device('cpu'))
model.to(device)

data_dir = os.path.join(path, 'DataSet', 'val')
test_transforms = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
                ])


def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    #input = Variable(image_tensor)
    input = image_tensor.to(device)
    output = model(input)
    #print(output)
    index = output.data.cpu().numpy().argmax()
    #index = output.data.numpy().argmax().to(device)
    #print(index)
    return index



def get_random_images(num):
    global classes
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
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



def predict_all_():
    count = 0
    to_pil = transforms.ToPILImage()

    images, labels = get_random_images(32)

    if os.path.exists(path + '/Scripts/class_list_updated2.txt'):
        with open(path + '/Scripts/class_list_updated2.txt', 'r') as f:
            lines = f.readlines()
        
        lines = str(lines)
        dict_lines = eval(lines)

        fig=plt.figure(figsize=(20,20))

        for ii in range(len(images)):
            image = to_pil(images[ii])
            index = predict_image(image)
            print("Ground Truth : ", str(labels[ii]), ", Prediction : ", index)
            if index != labels[ii]:
                count=count + 1

                truth = dict_lines[labels[ii]]
                pred = dict_lines[index]

                ground_truth = "True Label: " + str(truth)
                model_prediction = "Model_ Pred: " + str(pred)

                plt.title(ground_truth+" "+ model_prediction)
                plt.imshow(image)

                plt.show()
        print('Count for wrong Predictions: ', count)


if __name__ == '__main__':
    predict_all_()
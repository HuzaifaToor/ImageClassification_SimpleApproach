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

test_batch_size = 16
#torch.load with map_location=torch.device('cpu')
path = '/home/iuna/IUNA_AI/Classification/imagenet_myApproach'
device = torch.device('cpu')
model = torch.load(path + '/output/models/model_202209192220.pth', map_location=torch.device('cpu'))
model.to(device)

data_dir = os.path.join(path, 'DataSet', 'val')
test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224,224))
                ])


def predict_image(image):
    image_tensor = test_transforms(image).float()
    #image_tensor = image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.view(1, 3, 224, 224)
    with torch.no_grad():
        input = image_tensor.to(device)
        output = model(input)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, class_ = torch.max(probs, 1)
        model_pred = output.data.cpu().numpy().argmax()

    return model_pred, conf



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
    worng_pred_count = 0
    to_pil = transforms.ToPILImage()

    images, labels = get_random_images(16)
    print("Plotting False Predictions!!!\n")

    #Reading the class label names
    if os.path.exists(path + '/Scripts/class_names.txt'):
        with open(path + '/Scripts/class_names.txt', 'r') as f:
            lines = f.readlines()
        
        lines = str(lines)
        dict_lines = eval(lines)

        #iterate over the batch_size for prediction

        for ii in range(len(images)):
            image = to_pil(images[ii])
            model_pred, conf = predict_image(image)
            print("Ground Truth : ", str(labels[ii]), ", Prediction : ", index)
            if model_pred != labels[ii]:
                worng_pred_count=worng_pred_count + 1

                truth_labelName = dict_lines[labels[ii]]
                pred_labelName = dict_lines[model_pred]

                ground_truth = "True Label: " + str(truth_labelName)
                model_prediction = "Model_ Pred: " + str(pred_labelName)
                pred_score = str("Pred_Score: " + str('%.2f' % float(conf)))

                plt.title(ground_truth+ " " + model_prediction + " " + pred_score)
                plt.imshow(image)
                plt.show()

        print('Count for wrong Predictions: ', worng_pred_count)


if __name__ == '__main__':
    predict_all_()
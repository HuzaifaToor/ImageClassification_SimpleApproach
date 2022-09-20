import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from functools import partial
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
from torch.utils.data import DataLoader, Subset
from predict import *
from train import *
from CM import *


# current date and time
datetime = ''.join(item for item in str(datetime.now()) if item.isalnum())

#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

########## Parser Args #####################
parser = argparse.ArgumentParser(description='Image Classification')

parser.add_argument('-p', '--path', type=str, metavar='', default = '/home/iuna/IUNA_AI/Classification/imagenet_myApproach', help = 'Path to directory')
parser.add_argument('-lr', '--learning_rate', type = float, metavar='', default=0.001, help = 'Learning rate for Model')
parser.add_argument('-mt', '--momentum', type = float, metavar='', default=0.9, help = 'Momentum for SGD optimizer (Default = 0.9)')
parser.add_argument('-bs', '--batch_size', type=int, metavar='', default=32, help = 'Batch Size for Train and Validate Dataset')
parser.add_argument('-e', '--epochs', type=int, metavar='', default=5, help = 'Number of epochs for training')
parser.add_argument('-m', '--model_name', type=str, metavar='', default='resnet50', help = 'Training Model Name ')
parser.add_argument('-lg', '--log_folder', type=str, metavar='', default = 'logs_' + datetime[:12], help = 'Evaluation Logs Saving Directory')
parser.add_argument('-md', '--model_directory', type=str, metavar='', default= 'model_' + datetime[:12], help= 'Trained Model save Directory')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 2)')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

args = parser.parse_args()

##################################

def acc_loss_plots(history):

    history = np.array(history)
    plt.subplot(121)
    plt.plot(history[:,0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.xlim(0,args.epochs)
    plt.ylim(0,1)
    plt.subplot(122)
    plt.plot(history[:,2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.xlim(0,args.epochs)
    plt.ylim(0,1)
    #plt.savefig(dataset+'_accuracy_curve.png')
    plt.show()


def func_dataloader(data_path, batch_size, load_data = 'train'):

    train_directory = os.path.join(data_path + '/DataSet/train')
    valid_directory = os.path.join(data_path + '/DataSet/val')
    #test_directory = os.path.join(data_path + '/DataSet/test')

    if load_data == 'train':

        train_dataset = datasets.ImageFolder(
            train_directory,
            transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    #std=[0.229, 0.224, 0.225])
            ]))

        return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) , len(train_dataset), len(train_dataset.classes)

    elif load_data =='valid':

        valid_dataset = datasets.ImageFolder(
            valid_directory,
            transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
                ]))

        return torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False), len(valid_dataset), len(valid_dataset.classes)
    
    # else:

    #     test_dataset = datasets.ImageFolder(
    #         test_directory,
    #         transforms.Compose([
    #             transforms.Resize((224,224)),
    #             transforms.ToTensor()
    #             ]))

        return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False), len(test_dataset), len(test_dataset.classes)




if __name__ == '__main__':

    ########### Training and Validation Loader #############

    train_loader, train_data_size, num_classes = func_dataloader(args.path, batch_size = args.batch_size, load_data = 'train')
    valid_loader, valid_data_size, num_classes = func_dataloader(args.path, batch_size = args.batch_size, load_data = 'valid')
    
    ############### Model Function call ###################
    # model, optimizer, loss_criterion = model_(args.model_name,
    #     num_classes, 
    #     momentum=args.momentum, 
    #     weight_decay=args.weight_decay, 
    #     lr=args.learning_rate
    #     )
    
    ngpus_per_node = torch.cuda.device_count()

    model, optimizer, criterion = model_(args.model_name, 
                                num_classes, 
                                args.gpu, 
                                args.batch_size, 
                                args.learning_rate, 
                                args.momentum, 
                                args.weight_decay,
                                args.workers,
                                ngpus_per_node, 
                                pretrained=True, 
                                dist=False
                                )
    








    ############# Training Function  call ##################
    trained_model, history, best_epoch = train_and_validate(
        model, 
        criterion, 
        optimizer,
        train_loader,
        train_data_size,
        valid_loader,
        valid_data_size,
        num_classes,
        data_path = args.path,
        epochs = args.epochs,
        logs_folder = args.log_folder, 
        model_folder = args.model_directory,
        )

  
    ####### acc_loss_plots(history)

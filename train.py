#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 01:35:21 2022

@author: huzaifa
"""

import torch
import os
import time
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from model_arch import *
from CM import *
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################################################################################################
'''Train Function '''
########################################################################################################################################

def train_and_validate(
        model, 
        loss_criterion, 
        optimizer,  
        train_loader,
        train_data_size,  
        valid_loader,
        valid_data_size,  
        num_classes,
        data_path, 
        epochs, 
        logs_folder, 
        model_folder):
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param data_path: Directory to the dataset on local drive
        :param epochs: Number of epochs (default=20)
        :batch_size: number of images to be processed together
        :param logs_folder: Directory to save tensorboard log files
        :param model_folder: directory to save trained model
  
    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''
    
    start = time.time()
    history = []
    best_loss = 100000.0
    best_epoch = None

    ##############################################

    # If folder doesn't exist, then create it.
    if not os.path.isdir(data_path + 'output'):
        os.makedirs(data_path + '/output')
        os.makedirs(data_path + '/output/logs')
        os.makedirs(data_path + '/output/logs/'+logs_folder)
        os.makedirs(data_path + '/output/models')
        os.makedirs(data_path + '/output/models/'+model_folder)


    else:
        if not os.path.isdir(data_path + '/output/logs'):
            os.makedirs(data_path + '/output/logs')
            os.makedirs(data_path + '/output/logs/'+logs_folder)

        if not os.path.isdir(data_path + '/output/models'):
            os.makedirs(data_path + '/output/models')
            os.makedirs(data_path + '/output/models/'+model_folder)

    ###################### Summary Writer for tensorboard logs ###################
    writer = SummaryWriter(data_path + '/output/logs/'+logs_folder)
    
    ##############################################################
    for epoch in range(epochs):
        epoch_start = time.time()
        #print("Epoch: {}/{}".format(epoch+1, epochs))
        
        # Set to training mode
        model.train()

        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        train_best_acc = 0.0
        train_best_loss=10000.0
        
        valid_loss = 0.0
        valid_acc = 0.0
        valid_best_acc = 0.0
        valid_best_loss=10000.0

        correct_counts = 10.0

        print(f"\nEpoch [{epoch}]")

        
        for i, (inputs, labels) in enumerate(pbar :=tqdm(train_loader, colour="green")):

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Clean existing gradients
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            
            # Compute loss
            loss = loss_criterion(outputs, labels)
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            if loss.item()<train_best_loss:
                train_best_loss=loss.item()
                
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            if acc.item()>train_best_acc:
                train_best_acc=acc.item()
            
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            
            # show data on Progress bar 
            pbar.set_description(f"Training Progress  ")
            pbar.set_postfix(loss=train_best_loss, acc=train_best_acc)
        

        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(pbar2 := tqdm(valid_loader, colour="blue")):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)
                
                if loss.item()<valid_best_loss:
                    valid_best_loss=loss.item()

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                if acc.item()>valid_best_acc:
                    valid_best_acc=loss.item()
                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            
                pbar2.set_description(f"Validation Progress ")
                pbar2.set_postfix(loss=valid_best_loss, acc=acc.item())



        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch

        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_data_size 
        avg_train_acc = train_acc/train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/valid_data_size 
        avg_valid_acc = valid_acc/valid_data_size

        ''' Saving evaluation data to losg for tensor board'''
        
        writer.add_scalar(tag='Training_Loss', scalar_value=avg_train_loss, global_step=epoch)
        writer.add_scalar(tag='Training_Accuracy', scalar_value=avg_train_acc, global_step=epoch)

        writer.add_scalar(tag='Validation_Loss', scalar_value=avg_valid_loss, global_step=epoch)
        writer.add_scalar(tag='Validation_Accuracy', scalar_value=avg_valid_acc, global_step=epoch)

        #writer.add_scalar(tag='Correct_preds', scalar_value=correct_counts, global_step=epoch)

        # Saving evaluation data for simple plots
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
                
        epoch_end = time.time()

    
        #print("Epoch : {:03d}, \nTraining: Loss - {:.4f}, Accuracy - {:.4f}%, \nValidation : Loss - {:.4f}, Accuracy - {:.4f}%, Time: {:.4f}s\n\n\n".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        print("\nTraining: Average Loss - {:.4f}, Average Accuracy - {:.4f}%, \nValidation :Average Loss - {:.4f}, Average Accuracy - {:.4f}%, Time: {:.4f}s\n\n\n".format(avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        
        ############# Confusion matrix in tensor board ################
        #writer.add_figure("Confusion matrix", createConfusionMatrix(train_loader, model, num_classes), epoch)


        # Save if the model has best accuracy till now
        
    torch.save(model, data_path+ '/output/models/'+model_folder+'.pth')
    return model, history, best_epoch

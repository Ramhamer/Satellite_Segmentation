import torch
import wandb
import cv2
import io
import numpy as np
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import segmentation_models_pytorch as smp
import torch.nn as nn   
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import random
import torchvision.transforms as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from utils.image_utils import one_hot , grey_to_rgb_mask, add_mask
from utils.data_utils import get_directories
from colorama import Fore, Style, init
import glob
import json
from sklearn.metrics import precision_recall_curve
from utils.loss_utils import DiceCrossEntropyLoss, JaccardFocalLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_accuracy(predictions,masks):
    """
    Calculate the accuracy of the model predictions.

    Args:
        predictions (torch.Tensor): The model predictions with shape (N, H, W) or (N, C, H, W),
                                    where N is the batch size, C is the number of classes, H and W are the height and width of the masks.
        masks (torch.Tensor): The ground truth masks with the same shape as predictions.
        desirable_class (int): The class index for which to compute accuracy.

    Returns:
        float: The accuracy of the model predictions.
    """
    with torch.no_grad():
        # Get predicted class indices
        preds = predictions.argmax(dim=1)  # Shape: [B, H, W]
        
        # Prepare masks - remove channel dimension
        masks = masks.squeeze(1)  # Shape: [B, H, W]
        
        # Calculate accuracy
        accuracy = (preds == masks).float().mean().item()
        
    return accuracy

def iou_score(outputs, masks):
    """
    Calculate the mean Intersection over Union (IoU) for multi-class segmentation.

    Args:
        outputs (torch.Tensor): The model predictions (N, C, H, W) or (N, H, W).
        masks (torch.Tensor): The ground truth segmentation masks (N, H, W).

    Returns:
        float: The mean IoU across all classes.
    """
    #check unique values
    desirable_class =  np.union1d(np.unique(masks), np.unique(outputs))
    ious = []
    for cls in list(desirable_class):
        pred_cls = (outputs == cls)  # Predicted mask for class `cls`
        mask_cls = (masks == cls)    # Ground truth mask for class `cls`
        intersection = np.logical_and(pred_cls, mask_cls).sum().astype(float)

        union = np.logical_or(pred_cls, mask_cls).sum().astype(float)


        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)

    return np.mean(ious)


def get_class_one_accuracy(predictions,masks,num_cls):
    """
    Calculate the Intersection over Union (IoU) for a specific class.

    Args:
        predictions (torch.Tensor): The model predictions with shape (N, H, W) or (N, C, H, W),
                                    where N is the batch size, C is the number of classes, H and W are the height and width of the masks.
        masks (torch.Tensor): The ground truth masks with the same shape as predictions.
        num_cls (int): The class index for which to compute the IoU.

    Returns:
        float: The IoU for the specified class. If no area exists (union is 0), the IoU will be 0.0.
    """
    
    # Ensure the predictions and labels are on the same device
    predictions = predictions.to(masks.device)
    # Convert predictions to binary (0 or 1) if they are probabilities
    if predictions.dim() > 1 and predictions.size(1) > 1:
        predictions = predictions.argmax(dim=1)
        masks = masks.argmax(dim=1)
    # Initialize variables to keep track of intersection and union
    iou = 0.0
    pred_mask = ((predictions) == num_cls)
    true_mask = (masks == num_cls)
    
    intersection = (pred_mask & true_mask).sum().float()
    union = (pred_mask | true_mask).sum().float()
    
    if union > 0:
        iou += intersection / union
    
    if type(iou) != float :
        return iou.item()
    else:
        return iou

def validate_model(model, val_loader, criterion, device ,desirable_class):
    """
    Evaluate the model on the validation dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): The loss function used to compute the loss.
        device (torch.device): The device (CPU or GPU) to run the model on.
        desirable_class (int): The class index for which to compute accuracy.

    Returns:
        tuple: (val_loss, val_acc)
            - val_loss (float): Average loss over the validation dataset.
            - val_acc (float): Average accuracy for the desirable class over the validation dataset.
    """
    model.eval()
    loss_val = 0.0
    acc_val = 0
    total_val = 0
    building_acc = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            if type(outputs) == tuple:
                    outputs = outputs[0]
                    masks1 = masks
            if str(criterion) == "DiceLoss()":
                    masks=masks.argmax(dim=1)
             # Calculate accuracy and loss
            loss = criterion(outputs, masks)
            loss_val += loss.item() * images.size(0)
            acc_val += get_accuracy(outputs,masks,desirable_class)* images.size(0)
            building_acc +=  get_class_one_accuracy(outputs,masks,2)*images.size(0)
    val_loss = loss_val / len(val_loader.dataset)
    val_acc = acc_val/len(val_loader.dataset)
    building_acc = building_acc/len(val_loader.dataset)
    return val_loss ,val_acc
    
def load_model(cfg): 
    """""
    Initialize and return a model based on the specified configuration.

    Args:
        model_name (str): The name of the model architecture. Options include 'DeepLabV3Plus', 'Unet', 'PSPNet'.
        cfg

    Returns:
        torch.nn.Module: The initialized model instance.

    """

    model_name = cfg['model']['model_name']
    desirable_class = cfg['train']['desirable_class']
    if model_name == 'DeepLabV3Plus':
        encoder_weights = cfg['model']['encoder_weights']
        encoder_name = cfg['model']['encoder_name']
        activation = cfg['model']['activation']
        pooling = cfg['model']['pooling']
        dropout = cfg['model']['dropout']
        model = smp.DeepLabV3Plus(
            encoder_name= encoder_name, 
            encoder_weights= encoder_weights, 
            classes=desirable_class,
            activation=None,
            aux_params={
        'classes': desirable_class,
        'pooling': pooling,
        'dropout': dropout,}) 
    if model_name == 'Unet':
        model = smp.Unet(classes = desirable_class)
    if model_name == 'PSPNet':
        model = smp.PSPNet(classes = desirable_class)
    return model

def check_convergence(lst_loss,lst_loss_val,back_epochs,epslion):
    """
    Check if the model's training process has converged based on the loss values.

    Args:
        lst_loss (list of float): List of training loss values over epochs.
        lst_loss_val (list of float): List of validation loss values over epochs.
        back_epochs (int, optional): Number of recent epochs to consider for convergence checking. Default is 10.
        epsilon (float, optional): Threshold for determining negligible changes in loss. Default is 1e-4.

    Returns:
        bool: True if the model is considered to have converged, otherwise False.
    """
    # Ensure the list has at least 30 items
    if len(lst_loss) < 30:
        return False
    # Extract the last 10 items
    last_items = lst_loss[-back_epochs:]
    # Check if all items are equal or within the epsilon difference
    count = 0 
    for i in range(len(last_items) - 1):
            if abs(last_items[i] - last_items[i+1]) < epslion:
                count += 1
    if count == len(last_items)-1:
                return True  # Flag is not raised if any pair of items is not within the epsilon range
    count = 0
    # Check if margin between corresponding elements exceeds epsilon
    for a, b in zip(lst_loss[-10:], lst_loss_val[-10:]):
        if abs(a - b) >= 0.1:
            count +=1
    if count ==10:
            return True
    return False  # Flag is raised if all items are equal or within the epsilon range

def extract_values_between_strings(file_path, start_string, end_string):
    """
    Extract lines of text from a file that are located between two specific strings.

    Args:
        file_path (str): Path to the file to read.
        start_string (str): String that marks the beginning of the section of interest.
        end_string (str): String that marks the end of the section of interest.

    Returns:
        list of str: Lines of text found between the start and end strings.
    """
    with open(file_path, 'r') as file:
        content = file.read()

    # Initialize an empty list to store the values
    values = []

    # Split the content using the start string
    parts = content.split(start_string)
    if len(parts) > 1:
        # Further split the second part using the end string
        parts = parts[1].split(end_string)
        if len(parts) > 1:
            # Extract the values between the start and end strings
            values = parts[0].strip().split('\n')

    return values

def write_backup(train_accuracies, val_accuracies, train_losses,val_losses,model_name,train_dir):
    """
    Write training and validation metrics to a backup file.

    Args:
        train_accuracies (list of float): Training accuracies recorded over epochs.
        val_accuracies (list of float): Validation accuracies recorded over epochs.
        train_losses (list of float): Training losses recorded over epochs.
        val_losses (list of float): Validation losses recorded over epochs.
        model_name (str): Name of the model, used for naming the backup file.
        train_dir (str): Directory where the backup file will be saved.

    Returns:
        None
    """
    lst = [train_accuracies, val_accuracies, train_losses,val_losses]
    lists = {
    'train_accuracies': train_accuracies,
    'val_accuracies':val_accuracies,
    'train_losses':train_losses,
    'val_losses' : val_losses}
    file_path = os.path.join(train_dir, f'{model_name}_backup.txt')
    # Open a file in write mode ('w')
    with open(file_path, 'w') as f:
        for name, data_list in lists.items():
            # Write the name of the list to the file
            f.write(f'{name}:\n')
            
            # Write each item in the list to the file
            for item in data_list:
                f.write(f'{item}, ')
            
            # Add an empty line for separation
            f.write('\n')

def pad_to_mod16(arr):
    """
    Pad an array to ensure its dimensions (height and width) are multiples of 16.

    Args:
        arr (numpy array): Input array to be padded. Should have at least 2 dimensions.

    Returns:
        numpy array: Padded array with dimensions that are multiples of 16.
    """
    arr = np.array(arr)
    height, width = arr.shape[:2]
    pad_height = (16 - height % 16) % 16
    pad_width = (16 - width % 16) % 16

    padded_arr = np.pad(arr, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
    return padded_arr

def select_optimizer(model,optimizer_name,lr,weight_decay):
    """
    Select and initialize an optimizer for the given model based on the specified parameters.

    Args:
        model (torch.nn.Module): The model for which the optimizer is to be created.
        optimizer_name (str): Name of the optimizer to use (e.g., "AdamW").
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 regularization) factor for the optimizer.

    Returns:
        optim.Optimizer: Initialized optimizer object.
    """
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)   
        return optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)   
        return optimizer
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr,weight_decay=weight_decay)   
        return optimizer
    
def select_transform(config,test_mode = None):
    """
    Select and return image transformations for training and validation based on the given configuration.

    Args:
        config (dict): Configuration dictionary specifying which transformations to apply.
        test_mode (bool, optional): If True, return an empty list of transformations for validation. Defaults to None.

    Returns:
        tuple: A tuple containing two lists:
            - List of transformations to apply during training.
            - List of transformations to apply during validation (usually empty if test_mode is True).
    """
    if test_mode:
        train_transform = []
        return train_transform
    my_dict = { 'Rotation' :  T.RandomRotation(1),
    'horizontal_flip' :T.RandomHorizontalFlip(p=1), 'vertical_flip': T.RandomVerticalFlip(p=1),
    'ColorJitter':T.ColorJitter(brightness=random.uniform(0.1, 0.3), contrast=random.uniform(0.1, 0.3), saturation=random.uniform(0.1, 0.3), hue=random.uniform(0, 0.1))}
    
    train_transform = [my_dict[key] for key in my_dict if config['transformes']['types'][key] is True]
    
    return train_transform
 
def pick_best_model(test_dataset,models_list,model,train_dir,desirable_class):
    """
    Evaluates multiple pre-trained models on a test dataset and selects the best model 
    based on the highest main diagonal sum of the normalized confusion matrix.
    
    Args:
        test_dataset (Dataset): The dataset to evaluate the models on.
        models_list (list of str): List of model filenames or identifiers.
        model (torch.nn.Module): A PyTorch model instance.
        train_dir (str): Directory containing pre-trained model files.
        desirable_class (int): The class of interest that might influence class selection.
        device (torch.device, optional): The device to run the model on (GPU or CPU).
    
    Returns:
        str: The name of the best-performing model.
    """
    max_sum = 1
    best_model = None
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    class_list = ['no information','forest','buildings']
    if desirable_class == 2:
        class_list = class_list[:-1]
    bar_format = "{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)
    

    for model_name in tqdm(models_list, bar_format=bar_format):
        path = os.path.join(train_dir,model_name)
        model.load_state_dict(torch.load(path))
        model.to(device)
        model.eval()
        all_true_labels = []
        all_pred_labels = []
        
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, preds = torch.max(outputs, 1)  # Assuming outputs are logits
            
            all_true_labels.append(masks.argmax(dim=1).view(-1).cpu().numpy())
            all_pred_labels.append(preds.view(-1).cpu().numpy())
        
        all_true_labels = np.concatenate(all_true_labels)
        all_pred_labels = np.concatenate(all_pred_labels)
        cm = confusion_matrix(all_true_labels, all_pred_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm = np.array(cm_normalized)
        main_diagonal_sum = np.sum(np.diag(cm))
        if main_diagonal_sum>max_sum:
            best_model = model_name
            max_sum = main_diagonal_sum
    return best_model

def update_learning_curves(train_accuracy, val_accuracy, train_loss, val_loss, num_epochs, epoch, model_name,save_dir):
    """
    Update the learning curves for training and validation accuracy and loss.

    Args:
        train_accuracy (list): List of training accuracy values.
        val_accuracy (list): List of validation accuracy values.
        train_loss (list): List of training loss values.
        val_loss (list): List of validation loss values.
        num_epochs (int): Total number of epochs.
        epoch (int): The current epoch number.
        model_name (str): The name of the model.
        save_dir (str): The directory where the plot will be saved.

    Returns:
        None
    """
    global train_acc_line, val_acc_line, train_loss_line, val_loss_line
    if epoch == 0:
        # Initialize the plot only once
        plt.ion()
        plt.figure(figsize=(12, 6), num='Epoch Metrics')
        

        # Subplot for accuracy
        plt.subplot(1, 2, 1)
        train_acc_line, = plt.plot([], [], 'blue', label='Training Accuracy')
        val_acc_line, = plt.plot([], [], 'orange', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.suptitle(f'{model_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim(0, max(1,max(val_accuracy)))  
        plt.legend()
    
        # Subplot for loss
        plt.subplot(1, 2, 2)
        train_loss_line, = plt.plot([], [], 'blue', label='Training Loss')
        val_loss_line, = plt.plot([], [], 'orange', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()

    # Update the data for each line
    train_acc_line.set_data(range(1, len(train_accuracy) + 1), train_accuracy)
    val_acc_line.set_data(range(1, len(val_accuracy) + 1), val_accuracy)
    
    train_loss_line.set_data(range(1, len(train_loss) + 1), train_loss)
    val_loss_line.set_data(range(1, len(val_loss) + 1), val_loss)
 
    # Update the limits for the axes
    plt.subplot(1, 2, 1)
    plt.xlim(1, num_epochs)
    
    plt.subplot(1, 2, 2)
    plt.xlim(1, num_epochs)
    plt.ylim(0, max(1,max(val_loss)))  # Add a bit of margin to the y-axis

    # Remove old text annotations
    for ax in plt.gcf().get_axes():
        for txt in ax.texts:
            txt.remove()

    # Add new text annotations
    plt.subplot(1, 2, 1)
    plt.text(len(train_accuracy), train_accuracy[-1], f'{len(train_accuracy)}, {train_accuracy[-1]:.2f}', color='blue', weight='bold', fontsize=10, ha='right', va='bottom')
    plt.text(len(val_accuracy), val_accuracy[-1], f'{len(val_accuracy)}, {val_accuracy[-1]:.2f}', color='orange', weight='bold', fontsize=10, ha='right', va='bottom')

    plt.subplot(1, 2, 2)
    plt.text(len(train_loss), train_loss[-1], f'{len(train_loss)}, {train_loss[-1]:.2f}', color='blue', weight='bold', fontsize=10, ha='right', va='bottom')
    plt.text(len(val_loss), val_loss[-1], f'{len(val_loss)}, {val_loss[-1]:.2f}', color='orange', weight='bold', fontsize=10, ha='right', va='bottom')
    plt.draw()

    #plot the learning curves
    plt.pause(0.01)
    plt.show()
    
     # Save the plot
    save_path = os.path.join(save_dir,'learning_curves.png')
    plt.savefig(save_path)
    plt.close() 

    # Log to WandB
    # wandb.log({"Learning curves": wandb.Image(save_path)})
    
    return

def plot_confusion_matrix(best_model_dir,best_model, test_dataset,model,desirable_class): 
    """
    Plots and saves the confusion matrix for the best model on the test dataset.

    Args:
        best_model_dir (str): Directory containing the best model file.
        best_model (str): Name of the best model file.
        test_dataset (Dataset): The dataset to evaluate the model on.
        model (torch.nn.Module): A PyTorch model instance.
        desirable_class (int): The class of interest that might influence class selection.
        device (torch.device, optional): The device to run the model on (GPU or CPU).
    
    Returns:
        None
    """
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    class_list = ['no information','forest','buildings']
    if desirable_class == 2:
        class_list = class_list[:-1]
    path = os.path.join(best_model_dir,best_model)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, preds = torch.max(outputs, 1)  # Assuming outputs are logits
            
            all_true_labels.append(masks.argmax(dim=1).view(-1).cpu().numpy())
            all_pred_labels.append(preds.view(-1).cpu().numpy())
        
    all_true_labels = np.concatenate(all_true_labels)
    all_pred_labels = np.concatenate(all_pred_labels)

    cm = confusion_matrix(all_true_labels, all_pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm = np.array(cm_normalized)

    # Define class names if not already defined
    class_names = [f' {i}' for i in class_list]
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val:.2f}%', ha='center', va='center', color='red')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.title('Confusion Matrix (in %)')
    plt.savefig(best_model_dir)  
    plt.show()
    
def pad_to_mod16(arr):
    """
    Description:

    Args:
        
    Returns:
        
    """
    arr = np.array(arr)
    height, width = arr.shape[:2]
    pad_height = (16 - height % 16) % 16
    pad_width = (16 - width % 16) % 16

    padded_arr = np.pad(arr, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
    return padded_arr

def extract_values_between_strings(file_path, start_string, end_string=None):
    """
    Extracts numeric values between specified start and end strings in a file.

    The function reads the content of the file, looks for the section that starts with 
    `start_string`, and optionally ends with `end_string`. It parses the values in that 
    section, assuming they are separated by commas or new lines, and converts them to floats.

    Args:
        file_path (str): The path to the file to be read.
        start_string (str): The string that marks the beginning of the section from which values are to be extracted.
        end_string (str, optional): The string that marks the end of the section. If not provided, the function will extract everything after the start string until the end of the file.

    Returns:
        list of float: A list containing the extracted numeric values as floats.

    Raises:
        ValueError: If the values cannot be converted to float.
        
    """
    with open(file_path, 'r') as file:
        content = file.read()

    # Initialize an empty list to store the values
    values = []

    # Split the content using the start string
    parts = content.split(start_string)
    if len(parts) > 1:
        if end_string:
            # Further split the second part using the end string
            parts = parts[1].split(end_string)
            if len(parts) > 1:
                # Extract the values between the start and end strings
                values_str = parts[0].strip()
        else:
            # If end_string is not provided, take everything after start_string till the end
            values_str = parts[1].strip()
        
        # Split the extracted string into values
        for x in values_str.split('\n'):
            # Handle each line separately if needed
            for y in x.split(','):
                if len(y) != 0:
                    try:
                        values.append(float(y))
                    except ValueError:
                        print(f"Warning: '{y}' is not a valid float value.")
    return values

def process_and_plot(save_dir):
    """
    Processes a text file containing training and validation metrics, 
    and generates plots for accuracies and losses over epochs.

    The function looks for a `.txt` file in the specified directory, extracts training and validation 
    accuracies and losses using `extract_values_between_strings`, and then plots these values.

    Args:
        save_dir (str): Path to the directory where the `.txt` file is located. Also used to save the plot image.

    Returns:
        None: The function saves the plot as 'training_curve.png' in `save_dir` and displays the plot.
    """
    dict = {'train_accuracies:' : [] ,'val_accuracies:' : [] ,'train_losses:': [] , 'val_losses:' : []}
    txt_file = glob.glob(os.path.join(save_dir, '*.txt'))[0]
    lst = list((dict.keys()))
    for i in range(len(lst)-1):
        dict[lst[i]] = extract_values_between_strings(txt_file,lst[i],lst[i+1])
    dict[lst[-1]] = extract_values_between_strings(txt_file,lst[-1])
    train_acc = dict[lst[0]]
    val_acc = dict[lst[1]]
    train_loss = dict[lst[2]]
    val_loss = dict[lst[3]]
    epochs = range(1, max(len(train_acc), len(val_acc), len(train_loss), len(val_loss)) + 1)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Subplot 1: Accuracies
    if train_acc:
        ax1.plot(epochs[:len(train_acc)], train_acc, 'b', label='Training Accuracies')
    if val_acc:
        ax1.plot(epochs[:len(val_acc)], val_acc, 'orange', label='Validation Accuracies')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracies')
    ax1.set_title('Accuracies over Epochs')
    ax1.legend()

    # Subplot 2: Losses
    if train_loss:
        ax2.plot(epochs[:len(train_loss)], train_loss, 'b', label='Training Losses')
    if val_loss:
        ax2.plot(epochs[:len(val_loss)], val_loss, 'orange', label='Validation Losses')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Losses')
    ax2.set_title('Losses over Epochs')
    ax2.legend()

    # Adjust layout and display
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curve'), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()
 
def peek_version(version):
    """
    Returns the directory path for a given model version.

    Args:
        version (int): The version number of the model directory to retrieve.

    Returns:
        str: The directory path corresponding to the specified version.
    """
    parent_path = '/home/oury/Documents/Segmentation_Project/models'
    dir = parent_path
    if version == 1:
        dir = os.path.join(parent_path,'BCEWithLogitsLoss_landcover_crop_256')
    if version == 2:
        dir = os.path.join(parent_path,"BCEWithLogitsLoss_landcover_crop_512")
    if version == 3:
        dir = os.path.join(parent_path,"BCEWithLogitsLoss_landcover_compress_256")
    if version == 4:
        dir = os.path.join(parent_path,"BCEWithLogitsLoss_landcover_compress_512")
    if version == 5:
        dir = os.path.join(parent_path,"DiceLoss_landcover_crop_256")
    if version == 6:
        dir = os.path.join(parent_path,"DiceLoss_landcover_crop_512")
    if version == 6:
        dir = os.path.join(parent_path,"DiceLoss_landcover_compress_512")

    return dir

def convert_numpy_to_native(obj):
    """Recursively convert NumPy types to native Python types."""
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in 'biufc':  # If it's a numeric array
            return obj.tolist()  # Convert to list of native Python types
        else:
            return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16)):
        return int(obj)  # Convert NumPy int to Python int
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)  # Convert NumPy float to Python float
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(x) for x in obj]
    else:
        return obj

def vector_map(mask,output_path):
    """
    Converts a segmentation mask into vectorized polygons and saves the result in GeoJSON format.

    Args:
        mask (numpy.ndarray or torch.Tensor): The segmentation mask. The mask can be either a NumPy array or a PyTorch tensor.
        output_path (str): The file path where the resulting GeoJSON file will be saved.

    Returns:
        None: The function writes the GeoJSON file to the specified output path.

    Steps:
        - If the mask is a tensor, convert it to a NumPy array.
        - For each unique class in the mask, find contours.
        - Simplify the contours and convert them to a GeoJSON-compliant format.
        - Write the GeoJSON data to the specified output file.
    """
    if mask.__class__.__name__ == 'Tensor':
        mask = mask.cpu()
    mask = np.array(mask)
    if len(mask.shape) >2 :
        mask = np.transpose(mask, (1, 2, 0))
    unique_class = list(np.unique(mask))
    contours_per_class = {}
    features = []
    for cls in unique_class:
        # Step 1: Create a binary mask for the current class
        class_mask = (mask == cls).astype(np.uint8) * 255
         # Step 2: Find contours in the binary mask
        contours, hierarchy = cv2.findContours(class_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Store contours
        contours_per_class[cls] = contours
    
        for contour in contours:
            # Simplify the contour
            epsilon = 0.000000001 * cv2.arcLength(contour, True)
            simplified_contour = cv2.approxPolyDP(contour, epsilon, True)

            # Convert contour points to GeoJSON polygon format
            polygon = np.squeeze(simplified_contour).tolist()
            if len(polygon) < 3:
                continue

            if polygon[0] != polygon[-1]:
                polygon.append(polygon[0])# Ensure the polygon is closed

             # If lst is a list of lists, apply the conversion recursively
            
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon]
                },
                "properties": {
                    "class": str(cls)
                }
            }
            features.append(feature)
    
    features = convert_numpy_to_native(features)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
       # Write GeoJSON data to file
    with open(output_path, 'w') as f:
        json.dump(geojson, f, separators=(',', ':'))
    return

def pred_map(model,best_model_dir,path):
    """
    Generates a vectorized GeoJSON map from a model's prediction on a given image.

    Args:
        model (torch.nn.Module): The trained segmentation model.
        best_model_dir (str): Path to the model checkpoint file.
        path (str): Path to the input image file.
        output_dir (str, optional): Directory to save the resulting GeoJSON file. If not provided, the file will be saved in the same directory as the input image.

    Returns:
        None: The function generates and saves the GeoJSON file corresponding to the predicted segmentation.
    """
    model.load_state_dict(torch.load(best_model_dir))
    model.to(device)
    model.eval()
    img_name = (path.split('/', -1)[-1])
    img_name = img_name.replace(".png", ".geojson")
    image_transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    test_image = Image.open(path)
    test_image = np.array(test_image)
    test_tensor = image_transforms(test_image).cuda()
    test_tensor = test_tensor.unsqueeze(0)
    test_output = (model(test_tensor))
    gs_output = test_output[0].argmax(dim=1)
    vector_map(gs_output,img_name)
    return
 
def compare_map_mask(mask, img, original_map):
    """test_dataset
    Plots three RGB images side by side with titles.
    
    Args:
        mask (np.ndarray): The first RGB image (mask).
        img (np.ndarray): The second RGB image (image).
        original_map (np.ndarray): The third RGB image (original map).
    """
    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot the mask
    axes[0].imshow(mask)
    axes[0].set_title('Mask')
    axes[0].axis('off')
    
    # Plot the image
    axes[1].imshow(img)
    axes[1].set_title('Image')
    axes[1].axis('off')
    
    # Plot the original map
    axes[2].imshow(original_map)
    axes[2].set_title('Original Map')
    axes[2].axis('off')
    
    # Display the plot
    plt.tight_layout()
    plt.show()

def build_banch_mark(cfg,model_dir,test_dataset):

    classes = ['Precision', 'Recall','F1']
    dict = {}
    list_dir = os.listdir(model_dir)
    for i in list_dir:
        model_path = os.path.join(model_dir,i)
        if 'Unet' in model_path:
            model = load_model('Unet',cfg)
        if model_path in 'PSPNet':
            model = load_model('PSPNet',cfg)
        else:
            model = load_model('DeepLabV3Plus',cfg) 
        model.load_state_dict(torch.load(model_path))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        model.to(device)
        model.eval()
        all_true_labels = []
        all_pred_labels = []

        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, preds = torch.max(outputs, 1)  # Assuming outputs are logits
                
                all_true_labels.append(masks.argmax(dim=1).view(-1).cpu().numpy())
                all_pred_labels.append(preds.view(-1).cpu().numpy())
            
        all_true_labels = np.concatenate(all_true_labels)
        all_pred_labels = np.concatenate(all_pred_labels)

        cm = confusion_matrix(all_true_labels, all_pred_labels)
        precision, recall, thresholds = precision_recall_curve(all_true_labels ,all_pred_labels)
        f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]
        dict[cm] = [precision, recall,f1_scores]

    models = list(dict.keys())
    bar_width = 0.2
    index = np.arange(len(classes))
    plt.figure(figsize=(4*len(models), 2*len(models)))
    for i in models:
        plt.bar(index + models.index(i)*bar_width, dict[i], bar_width, label=models[i])

    # Add labels and titles
    plt.ylabel('Score')
    plt.title('Benchmark Comparison of Segmentation Models (IoU)')
    plt.xticks(index + bar_width, classes)
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.savefig(model_dir)
    plt.show()
    plt.pause(0.1)
    return

def weight_tensor(train_annotations_dir , number_of_class):
    """
    Compute class weights based on pixel frequencies in the training annotations for the BCE loss function.

    Args:
        train_annotations_dir (str): Directory containing the mask images.
        number_of_class (int): Number of classes in the segmentation task.
        device (str or torch.device): The device to which the tensor should be moved. Default is 'cpu'.

    Returns:
        torch.Tensor: A tensor of shape (number_of_class, 1, 1) with the computed weights.
    """
# Directory containing your mask images
    total_pixels = 0
    class_counts = np.zeros(number_of_class)
    for filename in os.listdir(train_annotations_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Load image
            image_path = os.path.join(train_annotations_dir, filename)
            mask = Image.open(image_path).convert('L')  # Convert to grayscale
            mask = torch.tensor(np.array(mask))
            mask = mask.unsqueeze(0)
            mask = one_hot(mask,number_of_class)
            for i in range(len(class_counts)):
                class_counts[i] += mask[:, i, :, :].sum()
            total_pixels += mask.numel()
    # Get class frequencies
    for i in range(len(class_counts)):
        total_pixels_without_class = total_pixels - class_counts[i]
        class_counts[i] = total_pixels_without_class / class_counts[i]
    # set as tensor
    pos_weight = torch.tensor(class_counts, dtype=torch.float, device=device)
    return pos_weight

def calculate_class_distribution(cfg,dataset):
    class_counts = {}
    
    for idx in tqdm(range(len(dataset)), desc="Processing Masks distribution for class weights:"):
        mask = dataset[idx][1]
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
            
        unique, counts = np.unique(mask, return_counts=True)
        for cls, count in zip(unique, counts):
            if cls in class_counts:
                class_counts[cls] += count
            else:
                class_counts[cls] = count

    total_pixels = sum(class_counts.values())
    #sort the dictionary by key from 0 to n
    class_counts = {cls: class_counts.get(cls, 0) for cls in range(max(class_counts.keys()) + 1)}
    
    class_distribution = {cls: count / total_pixels for cls, count in class_counts.items()}
    cfg['train']['class_distribution'] = class_distribution

    return class_distribution


def calculate_class_weights(class_distribution, smoothing_factor=0.1):
    max_proportion = max(class_distribution.values())
    class_weights = []

    for cls in range(max(class_distribution.keys()) + 1):
        proportion = class_distribution.get(cls, 0)
        weight = max_proportion / (proportion + smoothing_factor)
        class_weights.append(weight)

    pos_weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return pos_weight


def model_predict(model, input):
    output = model(input)[0]
    output = F.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    return output

###########loss utils########

# Select the Loss method
def select_loss(cfg,criterion_name,data_dir,loss_mode,desirable_class,log_loss,from_logits,smooth,ignore_index,eps,train_loader):
    """
    Select and initialize a loss function based on the given parameters.

    Args:
        criterion_name (str): Name of the loss function to use ("DiceLoss" or "BCEWithLogitsLoss").
        data_dir (str): Base directory for the dataset, used to compute class weights if necessary.
        loss_mode (str): Mode for the Dice loss function (e.g., "binary" or "multiclass").
        desirable_class (int): Number of classes in the dataset.
        log_loss (bool): Whether to compute the logarithm of the Dice loss.
        from_logits (bool): Whether the input to BCEWithLogitsLoss is raw logits.
        smooth (float): Smoothing factor for the Dice loss to avoid division by zero.
        ignore_index (int): Label index to ignore in the Dice loss.
        eps (float): Small constant to avoid division by zero in the Dice loss.

    Returns:
        nn.Module: Initialized loss function object.
    """
    
    if criterion_name == "DiceLoss":
        new_criterion = smp.losses.DiceLoss(loss_mode,list(range(desirable_class)),log_loss,from_logits,smooth,ignore_index,eps)            

    if criterion_name == "CrossEntropyLoss":
        train_distribution = calculate_class_distribution(cfg,train_loader.dataset)
        class_weights = calculate_class_weights(train_distribution)
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    if criterion_name == "JaccardLoss":
        criterion = smp.losses.JaccardLoss(loss_mode,list(range(desirable_class)),log_loss,from_logits,smooth,eps)

    if criterion_name == "FocalLoss":
        criterion = smp.losses.FocalLoss(loss_mode,ignore_index)
    
    if criterion_name == "DiceCrossEntropyLoss":
        train_distribution = calculate_class_distribution(cfg, train_loader.dataset)
        class_weights = calculate_class_weights(train_distribution)
        criterion = DiceCrossEntropyLoss(loss_mode, weight=class_weights, dice_weight=0.5, ce_weight=0.5, log_loss=log_loss, from_logits=from_logits, smooth=smooth, ignore_index=ignore_index, eps=eps)

    if criterion_name == "JaccardFocalLoss":
        criterion = JaccardFocalLoss(loss_mode, jaccard_weight=0.5, focal_weight=0.5, ignore_index=ignore_index, eps=eps)

    return criterion

def segmented_image(pred_tensor,image,pred_image_path):
    '''
    This function takes the predicted tensor and the image array and save the segmented image
    pred_tensor : torch.Tensor : the predicted tensor
    image_array : np.array : the image array
    pred_image_path : str : the path to save the image
    return : None
    '''
    rgb_mask = grey_to_rgb_mask(pred_tensor)         
    output = add_mask(image,rgb_mask)

    return

def build_epoch_dir(epoch_dir):
        """"
        Create a directory for the current epoch to save model checkpoints and samples."
        """
        os.makedirs(epoch_dir, exist_ok=True)
        os.makedirs(os.path.join(epoch_dir,"sampels"), exist_ok=True)
        os.makedirs(os.path.join(epoch_dir,"sampels","pred_samples"), exist_ok=True)
        os.makedirs(os.path.join(epoch_dir,"sampels","gt_samples"), exist_ok=True)
        return




def save_confusion_matrix(y_true,y_pred,epoch_dir,desirable_class,cfg):

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # noemalized by column (how accurate the prediction for each class are)
    normalized_cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)

    # Get unique classes and sort them
    classes = []
    for i in range(desirable_class):
        classes.append(i)

    
    # Generate labels based on unique classes
    class_labels = [f'Class {c}' for c in classes]

    # Change labels into class names
    if cfg['data']['name'] == 'MiniFrance':
        label_map = {
            'Class 0': 'Landscape',
            'Class 1': 'Urban area',
            'Class 2': 'Forest',
            'Class 3': 'Unknown',
        }
        class_labels = [label_map[label] for label in class_labels]

    # Create the heatmap with dynamic labels
    plt.figure(figsize=(10, 8))
    sns.heatmap(normalized_cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(epoch_dir,'confusion_matrix.png'))

    plt.close()  # Close the figure to prevent display
    
    return cm

def save_image_samples(images,masks,outputs,epoch_dir):

    mean = np.array([0.5, 0.5, 0.5])  # Change to your dataset's mean
    std = np.array([0.5, 0.5, 0.5])   # Change to your dataset's std

# Convert tensor to NumPy and undo normalization

    pred_dir = os.path.join(epoch_dir,"sampels","pred_samples")
    gt_dir = os.path.join(epoch_dir,"sampels","gt_samples")
    index = 0
    for image, mask, output in zip(images, masks, outputs):
        image = image.cpu().permute(1, 2, 0).numpy()
        
        # Scale to [0, 1] range
        min_val = image.min()
        max_val = image.max()
        if max_val > min_val:  # Avoid division by zero
            image = (image - min_val) / (max_val - min_val)
        image = (image * 255).astype(np.uint8)
        output = (torch.argmax(output, dim=0))
        i_pred_dir = os.path.join(pred_dir,f"image_{index}.png")
        i_gt_dir = os.path.join(gt_dir,f"image_{index}.png")

        pred_segmented_image = segmented_image(output.unsqueeze(0),image,i_pred_dir)
        gt_segmented_image = segmented_image(mask,image,i_gt_dir)
        

        # pred_image = Image.fromarray(output)
        # pred_image.save(pred_image_path)
        index += 1
    return
    


    




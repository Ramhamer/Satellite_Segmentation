import wandb
import torch
import io
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from utils.image_utils import class_label
from utils.train_utlis import iou_score



def wandb_init(project_name, run_name):
    """
    Function to initialize wandb
    """
    wandb.login(key="21dc09403b9e610afeb413e953f851cb4a5f18f2")
    wandb.init(project=project_name, name=run_name)

    wandb.define_metric("loss", summary="min", step_metric="epoch")  # Minimize loss
    wandb.define_metric("accuracy", summary="max", step_metric="epoch")  # Maximize accuracy

    return


def wandb_learning_curves(epoch,num_epochs,train_accuracies,val_accuracies,train_losses,val_losses):
    """
    Function to log learning curves to wandb
    """

    # Method 1: Using a flat structure with line identifiers
    # This prepares data in the format needed for the vega_lite approach
    steps_flat = []
    values_flat = []
    line_keys = []

    for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        # Add train data point
        steps_flat.append(i+1)
        values_flat.append(train_loss)
        line_keys.append("train")
        
        # Add val data point
        steps_flat.append(i+1)
        values_flat.append(val_loss)
        line_keys.append("val")

    # Complete fields dictionary matching ALL template variables in the Vega spec
    fields = {
        # These are used in the transform filters and encodings
        "step": "step",            # Column name for x-axis data
        "lineVal": "lineVal",      # Column name for y-axis data
        "lineKey": "lineKey",      # Column name for line categories
        
        # These are used for titles
        "title": "Performance Metrics",  # Chart title
        "xname": "Epochs"                # X-axis label
    }

    # Make sure your table has exactly these column names
    vega_table = wandb.Table(columns=["step", "lineVal", "lineKey"])
    for i in range(len(steps_flat)):
        vega_table.add_data(steps_flat[i], values_flat[i], line_keys[i])

    loss_chart = wandb.plot_table(
        vega_spec_name="szwecdan-ben-gurion-university-of-the-negev/new_chart",
        data_table=vega_table,
        fields=fields
    )

    # Method 2: Accuracies
    # This prepares data in the format needed for the vega_lite approach
    steps_flat = []
    values_flat = []
    line_keys = []

    for i, (train_accuracy, val_accuracy) in enumerate(zip(train_accuracies, val_accuracies)):
        # Add train data point
        steps_flat.append(i+1)
        values_flat.append(train_accuracy)
        line_keys.append("train")
        
        # Add val data point
        steps_flat.append(i+1)
        values_flat.append(val_accuracy)
        line_keys.append("val")

    # Complete fields dictionary matching ALL template variables in the Vega spec
    fields = {
        # These are used in the transform filters and encodings
        "step": "step",            # Column name for x-axis data
        "lineVal": "lineVal",      # Column name for y-axis data
        "lineKey": "lineKey",      # Column name for line categories
        
        # These are used for titles
        "title": "Performance Metrics",  # Chart title
        "xname": "Epochs"                # X-axis label
    }

    # Make sure your table has exactly these column names
    vega_table = wandb.Table(columns=["step", "lineVal", "lineKey"])
    for i in range(len(steps_flat)):
        vega_table.add_data(steps_flat[i], values_flat[i], line_keys[i])

    accuracies_chart = wandb.plot_table(
        vega_spec_name="szwecdan-ben-gurion-university-of-the-negev/new_chart",
        data_table=vega_table,
        fields=fields
    )

    # Log using the registered vega_spec approach
    wandb.log({
        "Accuracy_Curves": accuracies_chart,
        "Loss_Curves": loss_chart,})

    return


def wandb_visualization_table(images,masks,predicition,epoch,filenames):
    """
    Function to log a table of images to wandb
    """
    
    class_dict = class_label()

    table = wandb.Table(columns=["ID","Images","Predictions","Ground Truth","iou"])
    
    for id, (img, mask,predicition) in enumerate(zip(images,masks,predicition)):
        
        #preprocess 
        img = (img.cpu().numpy()).transpose(1, 2, 0)
        predicition = (torch.argmax(predicition, dim=0).cpu().numpy())
        mask = mask.squeeze(0).cpu().numpy()

        #convert to wandb format
        image = wandb.Image(img)    
        pred_mask = wandb.Image(img,masks={"predictions": {"mask_data": predicition, "class_labels": class_dict}})
        gt_mask = wandb.Image(img,masks={"ground_truth": {"mask_data": mask, "class_labels": class_dict}})
        iou = iou_score(predicition, mask)

        table.add_data(filenames[id], image, pred_mask, gt_mask,iou)

    wandb.log({f"{epoch}": table})

    return

def wandb_confusion_matrix(cm,epoch,save_dir):
    
    # noemalized by column (how accurate the prediction for each class are)
    normalized_cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)

    # Get unique classes and sort them
    class_dict = class_label()
    class_labels = [class_dict[i] for i in range(len(class_dict))]

    
    # Create the heatmap with dynamic labels
    plt.figure(figsize=(10, 8))
    sns.heatmap(normalized_cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')


      # Save to file
    
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()

    # Log to wandb
    wandb.log({"Confusion Matrix": wandb.Image(save_path),"epoch": epoch})
    return
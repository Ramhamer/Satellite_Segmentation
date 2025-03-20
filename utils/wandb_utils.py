import wandb
import torch
from utils.image_utils import class_label



def wandb_init(project_name, run_name):
    """
    Function to initialize wandb
    """
    wandb.init(project=project_name, name=run_name)

    wandb.define_metric("loss", summary="min", step_metric="epoch")  # Minimize loss
    wandb.define_metric("accuracy", summary="max", step_metric="epoch")  # Maximize accuracy

    return


def wandb_learning_curves(epoch,train_accuracies,val_accuracies,train_losses,val_losses):
    """
    Function to log learning curves to wandb
    """
    
    if epoch == 0:
        wandb.log({"loss": wandb.plot.line_series(
                xs=[1],
                ys=[[train_losses[0]], [val_losses[0]]],
                keys=["train", "val"],
                title="Training and Validation Loss",
                xname="Epoch"),
            "accuracy": wandb.plot.line_series(
                xs=[1],
                ys=[[train_accuracies[0]], [val_accuracies[0]]],
                keys=["train", "val"],
                title="Training and Validation Accuracy",
                xname="Epoch")})
    else:
        wandb.log({"loss": wandb.plot.line_series(
                xs=[i for i in range(epoch)],
                ys=[train_losses, val_losses],
                keys=["train", "val"],
                title="Training and Validation Loss",
                xname="Epoch"),
            "accuracy": wandb.plot.line_series(
                xs=[i for i in range(epoch)],
                ys=[train_accuracies, val_accuracies],
                keys=["train", "val"],
                title="Training and Validation Accuracy",
                xname="Epoch")})
    
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
        iou ="remember to add the iou"

        table.add_data(filenames[id], image, pred_mask, gt_mask,iou)

    wandb.log({f"{epoch}": table})

    return
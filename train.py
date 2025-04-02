# Define the Training Loop
import torch
import numpy as np
import os
import sys
from tqdm.auto import tqdm
from tqdm import tqdm
# from colorama import Fore, Style, init
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.data_utils import train_dir, create_metadata, load_data
from utils.train_utlis import *
from utils.cfg_utils import load_yaml
from utils.gpus_utils import *
import wandb
from utils.wandb_utils import *
import datetime
torch.backends.cudnn.enabled = False
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"  


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Add two hours to the current time automatically
new_time = datetime.datetime.now() + datetime.timedelta(hours=2)
time = new_time.strftime("%m_%d-%H_%M")




def train(rank,cfg, world_size): #Pull all the vars from the config file
    #cfg
    data_dir = cfg['data']['dir']

    #train
    criterion_name = cfg['loss']['name']
    desirable_class = cfg['train']['desirable_class']
    batch_size = cfg['train']['batch_size']
    optimizer_name = cfg['train']['optimizer_name']
    lr = cfg['train']['lr']
    weight_decay = cfg['train']['weight_decay']
    epslion = cfg['train']['check_convergence_epslion']
    num_epochs = cfg['train']['num_epochs']
    back_epochs = cfg['train']['back_epochs']
    amp = cfg['train']['amp']
    #loss
    loss_mode = cfg['loss']['mode']
    log_loss = cfg['loss']['log_loss']
    from_logits = cfg['loss']['from_logits']
    smooth = cfg['loss']['smooth']
    ignore_index = cfg['loss']['ignore_index']
    eps = cfg['loss']['eps']

    #transform
    transform_dict = cfg['transformes']['types']

    #model
    model_name = cfg['model']['model_name']
    encoder_weights = cfg['model']['encoder_weights']
    encoder_name = cfg['model']['encoder_name']
    activation = cfg['model']['activation']
    pooling = cfg['model']['pooling']
    dropout = cfg['model']['dropout']
    prev_weights = cfg['model']['prev_weights']
    interval_save_epoch = cfg['train']['interval_save_epoch']

    #project name
    project_name = cfg['project']['name']
    run_name = cfg['project']['run_name']
####################################################################################
    if  world_size > 1:
        setup(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

    if run_name == "None":
        run_name = time

    #initlize the wandb run
    wandb.login(key='21dc09403b9e610afeb413e953f851cb4a5f18f2') 
    wandb_init(project_name,run_name)
   

    # Create a directory for the train
    save_dir = train_dir(model_name,criterion_name)
    checkpoints_dir = os.path.join(save_dir,'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # load the data
    train_loader, val_loader= load_data(cfg,desirable_class,batch_size,data_dir,rank,world_size,test_mode=None)

    # Initialize the model, loss function, and optimizer
    model = load_model(cfg)  

    optimizer = select_optimizer(model,optimizer_name,lr,weight_decay)
    criterion  = select_loss(cfg,criterion_name,data_dir,loss_mode,desirable_class,log_loss,from_logits,smooth,ignore_index,eps,train_loader)
    
    if prev_weights != "None":
        model.load_state_dict(torch.load(prev_weights))
        print(f"{Fore.GREEN}Model weights loaded successfully{Fore.RESET}")

    if prev_weights == "None":
        print(f"{Fore.RED}Previous Model weights not loaded{Fore.RESET}")
 
    create_metadata(cfg,save_dir)

    # Training loop
    num_iter = len(train_loader)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_acc = 0
    val_acc = 0
    best_epoch = 0
    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # wandb
    wandb.watch(model, log="all")
    
    #try amp
    if amp:
        scaler = torch.cuda.amp.GradScaler()

    
    bar_format = "{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)
    bar_format1 = "{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)
    with tqdm(total=num_epochs, desc="Training Progress",ncols=150, unit='epoch',bar_format=bar_format) as epoch_bar:
        for epoch in range(num_epochs):
            if world_size > 1:
                train_loader.sampler.set_epoch(epoch)
            loss_val = 0
            acc_val = 0    
            with tqdm(total=num_iter, desc="batch Progress",ncols=100  , unit='iter',bar_format=bar_format1) as iter_bar:
                
                # Training step
                model.train()
                acc_train= 0 
                batch_loss = 0.0
                for batch_idx, batch in enumerate(train_loader):
                    images, masks,_= batch
                    # to device
                    images, masks = images.to(device), masks.to(device)
                    
                    if amp:
                        with torch.cuda.amp.autocast():
                        # Forward pass
                            outputs = model(images)[0]
                            loss_masks = masks.squeeze(1).long()                    
                            loss = criterion(outputs,loss_masks)
                    else:
                        outputs = model(images)[0]
                        loss_masks = masks.squeeze(1).long()                    
                        loss = criterion(outputs,loss_masks)

                    # Calculate accuracy and loss of iter
                    item_accuracy = get_accuracy(outputs,masks)
                    
                    # # Backward pass and optimization
                    if amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    
                    else:
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    # Calculate accuracy and lose of epoch
                    batch_loss += loss.item() * images.size(0) # multiply batch loss by the size batch
                    acc_train += item_accuracy*images.size(0) # same as the loss
                    iter_bar.update(1) #update
                    torch.cuda.empty_cache()
            # Calculate the loss and accuracy of the epoch
            epoch_loss = batch_loss / len(train_loader.dataset) # divide the loss by all the data set
            epoch_acc = acc_train/len(train_loader.dataset)       # same as loss
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)

            # open a dir for epoch
            epoch_dir = os.path.join(save_dir,f'epoch_{epoch}')
            build_epoch_dir(epoch_dir)

            # sample batch samples
            number_val_batch = len(val_loader)
            random_batch_idx = np.random.randint(0,number_val_batch)


            # Validation step
            torch.cuda.empty_cache()

            cm = np.zeros((desirable_class,desirable_class))
            with torch.no_grad():

                for batch_idx, batch in enumerate(val_loader):
                    images, masks , filenames = batch
                    images, masks = images.to(device), masks.to(device)
                    with torch.cuda.amp.autocast():
                        # Forward pass
                        outputs = model(images)[0]
                        loss_masks = masks.squeeze(1).long()

                        
                        y_pred = torch.argmax(outputs,dim=1).cpu().numpy().flatten()
                        y_true = masks.cpu().numpy().flatten()
                        cm += confusion_matrix(y_true,y_pred,labels=range(desirable_class))
                    
                        # Calculate accuracy and loss of the validation
                        loss = criterion(outputs, loss_masks)
                        loss_val += loss.item() * images.size(0)
                        acc_val += get_accuracy(outputs,masks)* images.size(0)
                        val_loss = loss_val / len(val_loader.dataset)
                        val_acc = acc_val/len(val_loader.dataset)

                        # Save images samples
                        if batch_idx == random_batch_idx:
                            # save_image_samples(images,masks,outputs,epoch_dir)
                            wandb_visualization_table(images,masks,outputs,epoch,filenames)

            # # Save confusion matrix
            wandb_confusion_matrix(cm,epoch)
        


            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # print the processes
            sys.stdout.flush()
            print("\naccuracy train:" ,epoch_acc , "loss train:" ,epoch_loss , "\n" "accuracy val:" , val_acc," loss val:",val_loss)
            epoch_bar.update()
            

            # log the metrics
            wandb_learning_curves(epoch,train_accuracies,val_accuracies,train_losses,val_losses)
         

            #plot curves
            # update_learning_curves(train_accuracies, val_accuracies, train_losses,val_losses ,num_epochs,epoch ,model_name,save_dir)            
            
            # Save the best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                file_name = os.path.join(checkpoints_dir,f'{model_name}_{criterion_name}_best.pth')
                if os.path.exists(file_name):
                    os.remove(file_name)
                torch.save(model.state_dict(),file_name)
                cfg['test_evaluation']['best_model_weight'] = file_name
                
            # Save the model
            if epoch % interval_save_epoch == 0:
                # Save the model
                torch.save(model.state_dict(),os.path.join(checkpoints_dir,f'{model_name}__{criterion_name}_epoch_{epoch}.pth'))
    
            # if the training is converged , stop the training
            if check_convergence(train_losses,val_losses,back_epochs,epslion):
                break
    
    # Save the best model in the Metadata
    with open(os.path.join(save_dir, 'metadata.txt'), 'a') as file:
        file.write(f"\nThe best accuracy is: {best_acc} in epoch: {best_epoch}\n")

    #save the wandb
    wandb.finish()

    cleanup()
    return checkpoints_dir

if __name__ == "__main__":
    yaml_file = '/workspace/config.yaml'
    cfg = load_yaml(yaml_file)
    world_size = torch.cuda.device_count()  # Number of GPUs (2 in your case)
    if world_size > 1:
        mp.spawn(train, args=(cfg,world_size), nprocs=world_size, join=True)
    else:
        rank = 0
        train(cfg,world_size)


import yaml
from colorama import Fore, init
init()

def load_yaml(file_path):
    """
    Load a YAML file.

    Args:
        file_path (str): The path to the YAML file to be loaded.

    Returns:
        dict: The contents of the YAML file as a dictionary. 
              Returns an empty dictionary if the file is not found.
    
    """
    try:
        with open(file_path, 'rt') as f:
            cfg = yaml.safe_load(f.read())
            return cfg
    except FileNotFoundError:
        print(f"Error: YAML file {file_path} not found.")
        return {}

def save_yaml(file_path, data):
    """
    Save data to a YAML file.

    Args:
        file_path (str): The path to the file where the YAML data will be saved.
        data (dict): The data to be saved in YAML format.

    Returns:
        None
    """
    
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    print(f"Updated YAML file saved at {file_path}.")

def print_sum(config):
    """
    Print a summary of the configuration to the console.

    Args:
        config (dict): The configuration dictionary to be summarized.

    Returns:
        None
    """
    
    print(f"{Fore.GREEN}You are going to start training a new model with the following configuration:{Fore.RESET}")    
    print(f"{Fore.CYAN}- Architecture: {Fore.RESET}{config['model']['model_name']}")
    print(f"{Fore.CYAN}- Loss Function: {Fore.RESET}{config['loss']['name']}")
    print(f"{Fore.CYAN}- Optimizer: {Fore.RESET}{config['train']['optimizer_name']}")
    print(f"{Fore.CYAN}- Weight Decay: {Fore.RESET}{config['train']['weight_decay']}")
    print(f"{Fore.GREEN}The dataset you will use is:{Fore.RESET}")
    print(f"{Fore.CYAN}- Dataset name: {Fore.RESET}{config['data']['name']}")
    print(f"{Fore.CYAN}- Path: {Fore.RESET}{config['data']['dir']}")
    print(f"{Fore.GREEN}Your training loop  will be:{Fore.RESET}")
    print(f"{Fore.CYAN}- Number of epochs: {Fore.RESET}{config['train']['num_epochs']}")
    print(f"{Fore.CYAN}- Batch size: {Fore.RESET}{config['train']['batch_size']}\n")
    print(f"{Fore.RED}Starting the training loop..{Fore.RESET}")
    return


def data_and_trainloop_cfg(config):
    """
    Prompt user to update the dataset path, dataset name, batch size, and number of epochs.
        
    Args: config (dict): The configuration dictionary to be updated.
        
    Returns: None
    """
    # Update the dataset path and name
    print(f"{Fore.GREEN}The current dataset is: {config['data']['name']}{Fore.RESET}")
    change_dataset = input(f"{Fore.CYAN}Do you want to change the dataset? {Fore.RESET}\nY/n\n")
    if change_dataset == 'Y':
        new_dataset_path = input(f"{Fore.CYAN}Enter the new dataset path: {Fore.RESET}")
        new_dataset_name = input(f"{Fore.CYAN}Enter the new dataset name: {Fore.RESET}")
        config['data']['dir'] = new_dataset_path
        config['data']['name'] = new_dataset_name
        print(f"{Fore.GREEN}Dataset path and name updated successfully{Fore.RESET}")
    if change_dataset == 'n':
        print(f"{Fore.GREEN}No changes were made to the dataset{Fore.RESET}")
    
    # Ask the user for batch size and epochs
    while True:
        new_batch_size = input(f"{Fore.CYAN}Insert the Batch size that you want to use:{Fore.RESET} \n1. 16 \n2. 32 {Fore.YELLOW}(Default){Fore.RESET} \n3. 64 \n4. else: \n")
        if new_batch_size == "1":
            config['train']['batch_size'] = 16
            break
        if new_batch_size == "2":
            config['train']['batch_size'] = 32
            break
        if new_batch_size == "3":
            config['train']['batch_size'] = 64
            break
        if new_batch_size == "4":
            new_batch_user = input(f"{Fore.CYAN}Enter the new batch size: {Fore.RESET}")
            config['train']['batch_size'] = int(new_batch_user)
            break
        else:
            print(f"{Fore.RED}Invalid input.{Fore.RESET}")
    
    while True:
        new_epochs = input(f"{Fore.CYAN}Insert the number of epochs that you want to use:{Fore.RESET} \n1. 80 \n2. 120 {Fore.YELLOW}(Default){Fore.RESET} \n3. 160 \n4. else: \n")
        if new_epochs == "1":
            config['train']['num_epochs'] = 80
            break
        if new_epochs == "2":
            config['train']['num_epochs'] = 120
            break
        if new_epochs == "3":
            config['train']['num_epochs'] = 160
            break
        if new_epochs == "4":
            new_epochs_user = input(f"{Fore.CYAN}Enter the new number of epochs: {Fore.RESET}")
            config['train']['num_epochs'] = int(new_epochs_user)
            break
        else:
            print(f"{Fore.RED}Invalid input.{Fore.RESET}")

def update_cfg(config):
    """
    Prompt user to update the model architecture, loss function, optimizer, and weight decay value.

    Args:
        config (dict): The configuration dictionary to be updated.

    Returns:
        dict: The updated configuration dictionary.

    """
    print(f"{Fore.CYAN}Please select your desired operation by entering the corresponding number.{Fore.RESET}")

    # Ask the user for a model name
    while True:
        new_model = input(f"{Fore.CYAN}Architecture:{Fore.RESET} \n1. Deep Lab V3+ {Fore.YELLOW}(Default){Fore.RESET} \n2. UNet \n3. PSPNet \n4. Unet++ \n")
        if new_model == "1":
            config['model']['model_name'] = 'DeepLabV3Plus'
            break
        if new_model == "2":
            config['model']['model_name'] = 'UNet'
            break
        if new_model == "3":
            config['model']['model_name'] = 'PSPNet'
            break
        if new_model == "4":
            config['model']['model_name'] = 'UnetPlusPlus'
            break
        else:
            print(f"{Fore.RED}Invalid input.{Fore.RESET}")

    # Ask the user for a Loss function name
    while True:
        new_loss = input(f"{Fore.CYAN}Loss Function:{Fore.RESET} \n1. Dice Loss \n2. Cross Entropy Loss {Fore.YELLOW}(Default){Fore.RESET} \n3. Jaccard Loss \n4. Focal Loss \n5. Combined Dice Loss + Cross Entropy Loss \n6. Combined Jaccard Loss + Focal Loss \n")
        if new_loss == "1":
            config['loss']['name'] = 'DiceLoss'
            break
        if new_loss == "2":
            config['loss']['name'] = 'CrossEntropyLoss'
            break
        if new_loss == "3":
            config['loss']['name'] = 'JaccardLoss'
            break
        if new_loss == "4":
            config['loss']['name'] = 'FocalLoss'
            break
        if new_loss == "5":
            config['loss']['name'] = 'DiceCrossEntropyLoss'
            break
        if new_loss == "6":
            config['loss']['name'] = 'JaccardFocalLoss'
            break
        else:
            print(f"{Fore.RED}Invalid input.{Fore.RESET}")

    # Ask the user for an optimizer name
    while True:
        new_optimizer = input(f"{Fore.CYAN}Optimizer:{Fore.RESET} \n1. AdamW {Fore.YELLOW}(Default){Fore.RESET} \n2. Adam \n3. SGD \n")
        if new_optimizer == "1":
            config['train']['optimizer_name'] = 'AdamW'
            break
        if new_optimizer == "2":
            config['train']['optimizer_name'] = 'Adam'
            break
        if new_optimizer == "3":
            config['train']['optimizer_name'] = 'SGD'
            break
        else:
            print(f"{Fore.RED}Invalid input.{Fore.RESET}")

    # Ask the user for a Weight Decay value
    while True:
        new_weight_decay = input(f"{Fore.CYAN}Weight Decay value:{Fore.RESET} \n1. 0.0001 \n2. 0.00001 \n3. 0.000001 \n4. 0.04 {Fore.YELLOW}(Default){Fore.RESET} \n")
        if new_weight_decay == "1":
            config['train']['weight_decay'] = 0.001
            break
        if new_weight_decay == "2":
            config['train']['weight_decay'] = 0.0001
            break
        if new_weight_decay == "3":
            config['train']['weight_decay'] = 0.00001
            break
        if new_weight_decay == "4":
            config['train']['weight_decay'] = 0.04
            break
        else:
            print(f"{Fore.RED}Invalid input.{Fore.RESET}")
    print(f"{Fore.GREEN}Update configuration is completed{Fore.RESET}")
    
    data_and_trainloop_cfg(config)
    print_sum(config)
    return config

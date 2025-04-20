# <div align="center">Satellite Image Semantic Segmentation</div>

## Project Overview
This project focuses on semantic segmentation of aerial/satellite imagery for land cover classification. The model processes RGB satellite images (256x256x3) in PNG format to identify and segment different land cover features. The project evaluates various deep learning architectures (such as **DeepLabV3+**, **UNet**, etc.) to determine the most effective approach for high-resolution satellite image segmentation, with applications in urban, forest, water and other terrain features detection.

<details open>
  <summary><strong>Pre-trained project details:</strong></summary>
<br>
While the project structure is modular and can accommodate different architectures and datasets, this specific implementation includes:
  
* A trained model using the **DeepLabV3+** architecture
* Training performed on a land cover dataset (included in the repository)
* Optimization using **AdamW** optimizer
* Training with **CrossEntropy loss** function for semantic segmentation
* Our dataset class is: **7**
</details>

## Table of Contents
- [Getting Started: Important Requirements and Labeling, Project Directory Structure](#getting-started-important-requirements-and-labeling)
- [Data Organization and Optional Dataset](#data-organization-and-optional-dataset)
- [Installations](#installations)
- [Model Selection, Loss Functions, and Optimizers](#Model-selection-loss-functions-and-optimizers)
- [Usage](#usage)
  - [Config file](#config-file)
  - [How to run a training loop via train.py](#how-to-run-a-training-loop-via-train)
  - [How to run a training loop via main.py](#how-to-run-a-training-loop-via-main)
  - [Test Evaluations and Model Comparison](#test-evaluation-and-model-comparison)

  - [How to use "Inference" - Check the model weights on an image](#how-to-use-inference)
- [Pre-trained Model Results](#pre-trained-model-results)

## Getting Started: Important Requirements and Labeling
- Ensure that the images and masks is organized and as written in "Data organinzation and Optional Dataset".
- Ensure that eace image has a corresponding mask
- All images and masks should be of the same resolution (e.g., **256x256**).
- Images should be in formats such as **PNG** or **JPEG**, and masks should be **grayscale** images where each pixel corresponds to a class label (e.g., 0 for background, 1 for urban, 2 for forest, etc.).
- Ensure proper class configuration by following these steps:
  1. **Match Classes:**
     - The number of classes must match between your masks and config.yaml
     - The model supports either 7 classes (full) or 4 classes (reduced)
  2. **To Reduce Classes:**
     - Update the 'desirable_class' parameter in config.yaml
     - Modify the class mapping dictionary in image_utils.py under the class_reduction function
- **If you want to tag images for this semantic segmentation project, please visit [The CVAT Labeling Infrastructure Guide](https://github.com/danszwec/segmentation-labeling-Toolkit), where you'll find instructions on how to tag and save the images.**

### Project Directory Structure:
To set up the project environment:
1. Clone the repository to your desired location:
```bash
git clone https://github.com/danszwec/satellite_segmentation
cd satellite_segmentation
```
2. Navigate to your project directory and ensure it has the following structure:
```
your_model_path/
├── Data/
│   └── Dataset_dir/
├── result/          # Generated automatically
├── utils/
├── main.py
├── testEval.py
├── train.py
├── inference.py
└── config.yaml
```

## Data Organization and Optional Dataset
### Data Organization
The dataset should be organized in the following structure:

1. **`Dataset_dir/train/images/`**: Place your training images here.
2. **`Dataset_dir/train/masks/`**: Place the corresponding ground truth masks for training images here.
3. **`Dataset_dir/val/images/`**: Place your validation images here.
4. **`Dataset_dir/val/masks/`**: Place the corresponding ground truth masks for validation images here.
5. **`Dataset_dir/test/images/`**: Place your test images here.
6. **`Dataset_dir/test/masks/`**: Place the corresponding ground truth masks for test images here.

### Optional Dataset
To work with the same dataset used in this project, you can download it from https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset on Kaggle.

**Important Note:** Before using the dataset, you'll need to perform preprocessing steps as outlined in the data preparation section of this repository. The preprocessing is essential to ensure the data is in the correct format for the model training and evaluation processes.

## Installations:

To run this project, we recommend using our Docker environment which ensures consistent setup across different systems.

First, make sure you have Docker installed on your system, along with NVIDIA drivers and NVIDIA Container Toolkit installed and properly configured. Once these prerequisites are met, pull our PyTorch Docker image by running
```bash
docker pull nvcr.io/nvidia/pytorch:23.10-py3. 
```
After the image is downloaded, you can create and start your container using the following command:
```bash
docker run -it --gpus all --name pytorch_container -v /path/to/your/project:/workspace nvcr.io/nvidia/pytorch:23.10-py3
```
Make sure to replace **/path/to/your/project** with the actual path to your project directory on your local machine. This command creates a container with GPU support and mounts your project directory, allowing seamless file access between your host system and the container. Once inside the container, install the required dependencies by running
```bash
pip install -r requirements.txt.
```
You can verify your setup by checking GPU availability in Python with **torch.cuda.is_available()**, which should return **True** if everything is configured correctly.


## Model Selection, Loss Functions, and Optimizers
### Model Architecture
This project offers several segmentation models:
* DeepLabV3+: Uses atrous convolution and a decoder for precise boundary segmentation.
* UNet: Classic encoder-decoder model with skip connections, effective for medical image tasks.
* PSPNet: Uses pyramid pooling for multi-scale context and improved scene understanding.
* UNet++: Extends UNet with nested skip pathways for better feature fusion and accuracy.

### Loss Functions
Available loss functions include:
* Dice Loss: Optimizes for overlap between predicted and ground truth.
* BCEWithLogitsLoss: Combines binary cross-entropy with a sigmoid layer.
* Jaccard Loss: Focuses on maximizing intersection-over-union (IoU).
* Focal Loss: Down-weights well-classified examples to focus on hard ones.

### Optimizers
Choose from the following optimizers:
* AdamW: Optimizer with weight decay, ideal for most tasks.
* Adam: Adaptive optimization combining RMSProp and momentum.
* SGD: Classic optimizer, effective for fine-tuning.

### Default Configuration
The default setup uses **DeepLabV3+** with **CrossEntropy Loss** and the **AdamW** optimizer, offering a good balance for general segmentation tasks.


## Usage

### config File
The config.yaml file contains all the essential settings for configuring your training process. By modifying these settings, you can fine-tune the training process to suit your dataset and desired outcomes.

**Key Settings to Modify for Effective Training**
Before running the test, you can adjust the following parameters in the config.yaml to optimize your training:
- Model: Choose the architecture (e.g., DeepLabV3+, UNet, etc.).
- Loss Function: Select a suitable loss function (e.g., BCEWithLogitsLoss, Dice Loss).
- Optimizer: Choose between AdamW, Adam, or SGD.
- Learning Rate: Adjust the learning rate for optimal training speed and stability.
- Batch Size: Modify the batch size to suit your GPU memory capacity.
- Epochs: Set the number of epochs based on the size of your dataset and training goals.

**Make sure the settings align with your hardware capabilities and dataset size for efficient training.**

There are two options for modifying the config.yaml file:
1. Go to the config.yaml file, change the settings, and save it (this will become your default configuration).
2. Use the menu and choose not to use the default configuration (this will be a temporary configuration). If you like the outcome of the model after the run, you can set it as the default at the end.

### How to run a training loop via train

**In tain.py - Follow the prompts on-screen to ensure a smooth workflow.**
1. **Update Configurations:** Modify the config.yaml file with your desired parameters and ensure all paths and settings are correctly specified.
2. **In the buttom of train.py** - At the bottom of train.py, update the path to your config.yaml file
3. **Launch the Script:** Open your terminal or command line interface, navigate to the project directory and run the script using:
```bash
python train.py
```
### How to run a training loop via main
**In main.py - Follow the prompts on-screen to ensure a smooth workflow.**

1. **Launch the Script:** Open your terminal or command line interface, navigate to the project directory and run the script using:
   ```bash
   python main.py
   ```
2. **Welcome Screen:**
   - You will be presented with a welcome screen.
![image](https://github.com/user-attachments/assets/6453ad70-6760-41b7-887f-12f24a34e6a7)
   - Select 1 to initiate the model training process.

3. **Training Workflow Environment:**
   - You will enter the training workflow environment.
   - Choose whether to use the default pre-trained configuration:
     - Y: Load the default configuration.
     - n: Enter the configuration update screen.

**Configuration Options:** In the configuration update screen, customize the following:
- Architecture: Select from the following options:
  * DeepLabV3+
  * Unet
  * PSPNet
  * Unet++
- Loss Function: Choose one of the following:
  * Dice Loss
  * Cross Entropy Loss
  * Jaccard Loss
  * Focal Loss
- Optimizer: Select an optimizer:
  * AdamW
  * Adam
  * SGD
- Weight Decay: Choose a weight decay value:
  * 0.001
  * 0.0001
  * 0.00001
4. **Dataset Configuration:**
   - You will be asked if you want to change the dataset:
     - Y: Specify the path to your dataset and provide a name.
     - n: Continue using the current dataset.
5. **Additional Settings:**
   - Choose the batch size for training.
   - Specify the number of epochs.

After finalizing all configurations, you will see a summary of the selected settings, and the training loop will begin automatically.


## Test Evaluation and Model Comparison

At the end of the training process, the program automatically compares the performance of different checkpoints to help you evaluate the model's progress over time. This comparison provides insights into how the model's performance varies between different stages of training and across different architectures.

**If you want to run the test evaluation separately after getting your results, use this command:**  
```bash
python test_evaluation.py
```
**Important:** When running test_evaluation.py, you need to specify the path to the results directory. If you want to compare different models or batches, you can create a new directory containing all the weights and run the test evaluation on that directory.

### Metrics Evaluated
The program computes and reports the following metrics for each checkpoint:

- Recall
- Precision
- F1 Score
- Pixel Accuracy
- IoU (Intersection over Union)
- Inference Speed
These metrics provide a comprehensive evaluation of model performance and efficiency.


### How to use Inference
**Important! Before proceeding, You MUST verify the number of classes in your pre-trained weights and UPDATE the desired_class value in the config.yaml file accordingly**

There are two methods to run the inference module:
1. Using main.py
2. Using inference.py

**Method 1: Using main.py**
This method provides an interactive interface with on-screen prompts
1. **Launch the Script:** Open your terminal or command line interface, navigate to the project directory and run the script using:
   ```bash
   python main.py
   ```
2. **Welcome Screen:**
   - You will be presented with a welcome screen.
![image](https://github.com/user-attachments/assets/6453ad70-6760-41b7-887f-12f24a34e6a7)
   - Select 3 to apply an existing model for image segmentation.

3. **Inference Workflow Environment:**
   - You will enter the Pre-Trained Model Inference workflow environment.
   - Choose the architecture you want to evaluate.
   - Enter the path to the model weights you wish to use.
   - Provide the path to the image you want to predict.
   - Choose your output format:
     - **Press 1** for a segmented image.
     - **Press 2** for a vector map (GeoJSON)

**Method 2: Using inference.py**
This method offers a more direct approach:
1.  **Prepare Your Environment**:  have your image path and weight path ready inside inference.py script and update the config.yaml file if necessary.
2.  **Launch the Script:** Open your terminal or command line interface, navigate to the project directory and run the script using:
   ```bash
   python inference.py
   ```

## Pre-trained Model Results
Our semantic segmentation model was trained using a DeepLabV3+ architecture with CE loss function, processing 256x256 RGB aerial/satellite images. The model's performance exceeded our initial expectations, achieving a remarkable **90% accuracy on the training set and 70% on the validation set,** demonstrating strong generalization capabilities.
![6](https://github.com/user-attachments/assets/9e403a90-6bab-4dbe-9ced-e6a11dbc5e7b)

The training process showed consistent improvement, with both training and validation loss **converging to approximately 0.005.** This parallel reduction in loss values indicates that our model effectively learned the underlying patterns without overfitting to the training data. 

To demonstrate the model's practical effectiveness, we've included several example predictions below. These visualizations showcase the model's ability to accurately segment different land cover features from aerial imagery.

![grid_image](https://github.com/user-attachments/assets/1070e585-cc2e-4485-87b9-584c874380f1)


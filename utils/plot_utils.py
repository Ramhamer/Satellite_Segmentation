import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix_with_metrics(results, model_path, metric_path):
    """
    Function to plot the confusion matrix and display evaluation metrics in a fancy style above it.
    
    Parameters:
    - results: List containing confusion matrix and metrics
        - results[0]: Confusion matrix (2D NumPy array)
        - results[1]: Pixel accuracy (float)
        - results[2]: IoU (micro) (float)
        - results[3]: IoU (weighted) (float)
        - results[4]: Lowest IoU per class (float)
        - results[5]: Recall values per class (NumPy array)
        - results[6]: Precision values per class (NumPy array)
        - results[7]: F1 score values per class (NumPy array)
    - model_path: Path to save the evaluation metrics plot image (string)

    Returns:
    - None

    """

    #checkpoint name
    checkpoint_name = model_path.split('/')[-1]
    # Set the style of the plot
    sns.set(style="whitegrid")

    # Create the figure for the confusion matrix
    plt.figure(figsize=(12, 12))

    # Plot the confusion matrix
    sns.heatmap(results[0], annot=True, fmt='.5f', cmap='Blues', linewidths=0.5, linecolor='gray',
                cbar_kws={'label': 'Number of Pixels'}, annot_kws={"size": 8, "weight": "bold"})

    # Add title and labels
    plt.title('Confusion Matrix', fontsize=16, weight='bold')
    plt.xlabel('Predicted', fontsize=12, weight='bold')
    plt.ylabel('True', fontsize=12, weight='bold')

    # Prepare the metrics text in a clean format
    metrics_text = f"""
    Pixel Accuracy:        {results[1]:.4f}
    Recall:                {results[2]:.4f}
    Precision:             {results[3]:.4f}
    F1 Score:              {results[4]:.4f}
    """
    plt.subplots_adjust(top=0.75)  # Move everything up to give space for the metrics
    # Add the model name in cyan
    plt.gcf().text(0.43, 0.95, checkpoint_name, fontsize=15, weight='bold', color='cyan', ha='center')
     # Adjust the layout and place the text in the middle above the confusion matrix
    
    plt.gcf().text(0.43, 0.77, metrics_text, fontsize=14, weight='bold', color='darkblue', ha='center')

    # Save the figure
    path = os.path.join(metric_path,checkpoint_name + '_confusion_matrix.png')
    plt.savefig(path)    
    return  



def compare_models_performers(models_dict, metric_path):
    """
    Compare items in the lists of each key in the dictionary and display comparisons
    in the same bar graph, where each group represents the same index across models.

    Parameters:
    - models_dict: Dictionary where keys are model names/identifiers, and values are lists of metrics.
    - metric_path: Path to save the comparison plot image (string)
    Returns:
    - None
    """
    # Ensure the dictionary is not empty
    if not models_dict:
        print("The dictionary is empty!")
        return
    
    # for each value delete the first element
    for key in models_dict:
        models_dict[key] = models_dict[key][1:]

    # Validate that all lists in the dictionary are of the same length
    list_lengths = [len(v) for v in models_dict.values()]
    if len(set(list_lengths)) != 1:
        print("Error: All lists in the dictionary must have the same length.")
        return
    
    metrics_names = ['Pixel Accuracy', 'IoU (Micro)', 'IoU (Weighted)', 'Lowest IoU per Class', 'Recall', 'Precision', 'F1 Score','average time']

    # Extract the model names
    model_names = [key.split('/')[-1] for key in models_dict.keys()]  

    # Convert to a NumPy array for easier slicing
    metrics = np.array(list(models_dict.values()))

    # Set up GridSpec for custom layout
    fig = plt.figure(figsize=(25,14))  # Adjust figure size
    gs = gridspec.GridSpec(2, 4, figure=fig)  # 2 rows, 4 columns grid
    
    # Create axes
    axes = []
    for i in range(4):  # Top row (4 axes)
        axes.append(fig.add_subplot(gs[0, i]))
    for i in range(4):  # Bottom row (3 axes)
        axes.append(fig.add_subplot(gs[1, i]))

    # Plot the remaining metrics
    for i in range(len(metrics_names)):
        axes[i].bar(model_names, [metric[i] for metric in metrics], color='skyblue', alpha=0.8)
        axes[i].set_title(metrics_names[i], fontsize=10)
        axes[i].set_ylabel(f'Metric {i+1} Value', fontsize=7)
        axes[i].set_xlabel('Models', fontsize=6)
        axes[i].tick_params(axis='x', rotation=45,labelsize=10)

    # Adjust layout with more space between the rows
    plt.subplots_adjust(bottom=0.15, hspace=0.6, wspace=0.4)  # Increase hspace for vertical spacing
    
    # Adjust layout for better spacing
    plt.tight_layout()
    # Save the figure
    path = os.path.join(metric_path, 'models_comparison.png')
    plt.savefig(path)
    plt.close(fig)

    return



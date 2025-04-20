import torch
import os
import time
import numpy as np
import wandb
import datetime
from sklearn.metrics import jaccard_score, confusion_matrix
from utils.train_utlis import load_model, model_predict, calculate_class_distribution
from utils.data_utils import load_data
from utils.cfg_utils import load_yaml
from utils.wandb_utils import wandb_init, wandb_confusion_matrix
import pandas as pd

class SegmentationEvaluator:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.data_dir = cfg['test_evaluation']['dir']
        self.desirable_class = cfg['test_evaluation']['desirable_class']
        self.labels = [i for i in range(self.desirable_class)]

        ## Add two hours to the current time automatically
            

        #wandb init
        run_name = cfg['test_evaluation']['run_name']
        if run_name == "None":
            new_time = datetime.datetime.now() + datetime.timedelta(hours=2)
            time = new_time.strftime("%m_%d-%H_%M")
            run_name = time
        

        # Initialize wandb
        wandb_init('Evaluation',run_name)

      

    def evaluate_model(self, model_path, model):
        """Evaluate a single model and log metrics to wandb"""
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        model.eval()

        # Load test data
        _, test_loader = load_data(self.cfg, self.desirable_class, 1, self.data_dir, test_mode=True)
        test_counts = calculate_class_distribution(cfg,test_loader.dataset)

        predictions = []
        targets = []
        inference_times = []

        with torch.no_grad():
            for input, target,_ in test_loader:
                input = input.to(self.device)
                target = (target.squeeze(1)).cpu().numpy()
                target = target.flatten()

                # Measure inference time
                start_time = time.time()
                output = model_predict(model, input)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                output = output.cpu().numpy()
                output = output.flatten()
                predictions.append(output)
                targets.append(target)

        # Concatenate predictions and targets
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)

        # Calculate metrics
        metrics = self._calculate_metrics(predictions, targets)
        
        # Log metrics to wandb
        # self._log_metrics_to_wandb(metrics, os.path.basename(model_path))

        return metrics

    def _calculate_metrics(self, predictions, targets):
        """Calculate all relevant segmentation metrics"""
        metrics = []

        # Confusion Matrix
        conf_matrix = confusion_matrix(targets, predictions, labels=self.labels)
       
        # Normalized Confusion Matrix (row-wise normalization)
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        norm_conf_matrix = conf_matrix / row_sums
        metrics.append(norm_conf_matrix)

        # Pixel Accuracy
        true_positive = np.trace(conf_matrix)
        total_pixels = np.sum(conf_matrix)
        pixel_accuracy = true_positive / total_pixels
        metrics.append(pixel_accuracy)

        # IoU (Jaccard) scores
        iou_micro = jaccard_score(targets, predictions, average='micro', labels=self.labels)
        metrics.append(iou_micro)
        
        # Per-class IoU
        iou_per_class = []
        for label in self.labels:
            jaccard_class = jaccard_score(targets, predictions, average=None, labels=[label])[0]
            iou_per_class.append(jaccard_class)
            metrics.append(jaccard_class)
        
        return metrics

    def _log_metrics_to_wandb(self, metrics, model_name):
        """Log metrics to wandb"""
        # Create a dictionary for all metrics to log
        log_dict = {}
        
        # Log scalar metrics
        log_dict[f"{model_name}/pixel_accuracy"] = metrics['pixel_accuracy']
        log_dict[f"{model_name}/iou_micro"] = metrics['iou_micro']
        log_dict[f"{model_name}/avg_inference_time"] = metrics['avg_inference_time']
        
        # Log per-class IoU
        for label in self.labels:
            log_dict[f"{model_name}/iou_class_{label}"] = metrics[f'iou_class_{label}']
        
        # Log the metrics dictionary
        wandb.log(log_dict)
        

        # Log confusion matrix as a table for better visualization
        conf_matrix_table = wandb.Table(
            columns=["True Class", "Predicted Class", "Count"]
        )
        
        for i, true_class in enumerate(self.labels):
            for j, pred_class in enumerate(self.labels):
                conf_matrix_table.add_data(true_class, pred_class, metrics['normalized_confusion_matrix'][i, j])
        
        wandb.log({
            f"{model_name}/confusion_matrix_table": conf_matrix_table
        })




def evaluate_models(train_dir, cfg, device):
    """Evaluate multiple models and compare their performance"""
    # Initialize evaluator
    evaluator = SegmentationEvaluator(cfg, device)
    
    # Load model architecture
    model = load_model(cfg)
    
    # Get all model checkpoints
    model_paths = [os.path.join(train_dir, item) for item in os.listdir(train_dir) 
                  if os.path.isfile(os.path.join(train_dir, item))]
    
    # Evaluate each model
    results = {}
    for model_path in model_paths:
        print(f"Evaluating model: {model_path}")
        metrics = evaluator.evaluate_model(model_path, model)
        results[os.path.basename(model_path)] = metrics
    
    # Create comparison charts for all metrics
    create_comparison_charts(results, cfg)

    wandb.finish()
    return results

def create_comparison_charts(results, cfg):
    """Create comparison charts for all metrics across different models using wandb tables and plots"""
    # Create benchmark directory with today's date
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    benchmark_dir = os.path.join("benchmark", today)
    os.makedirs(benchmark_dir, exist_ok=True)
    
    metrics_names = ["normalized_confusion_matrix",  "pixel_accuracy", "mean_iou", "iou_per_class"]
    
    #make chart from every metric
    for index , metric in enumerate(metrics_names):
        # Extract metric values for each model
        data = []
        
        for model_path in results.keys():
            model_name = os.path.basename(model_path)
            if metric == "normalized_confusion_matrix":
                #implement it
                wandb_confusion_matrix((results[model_path])[index],model_name=model_name)

                # Skip confusion matrix as it's a 2D matrix
                continue
            elif metric == "iou_per_class":
                # need to implement it
                continue
            else:
                metric_value = (results[model_path])[index]
            data.append([metric_value,model_name])
    
            # Create wandb table
            table = wandb.Table(data=data, columns=[metric,"model_name"])
                    
            # Log the bar chart
            wandb.log({
                f"{metric}": wandb.plot.bar(
                    table, 
                    "model_name", 
                    metric, 
                    title=f"Comparison of {metric} across models"
                )
            })

    # Create per-class IoU comparison
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = 'models'
    yaml_file = '/workspace/config.yaml'
    cfg = load_yaml(yaml_file)
    evaluate_models(train_dir, cfg, device)
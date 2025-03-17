#!/bin/bash

# Script to update config values in YAML file and run training
# Usage: ./update_and_train.sh [data_dir] [data_name] [prev_weights] [batch_size] [class_distribution] [desirable_class] [lr] [num_epochs]

# Default values (current values from config)
CONFIG_FILE="config.yaml"
DATA_DIR="Rachel_Tzuria/Data/Dataset/Full_dataset"
DATA_NAME="MiniFrance"
PREV_WEIGHTS="None"
BATCH_SIZE=16
CLASS_DISTRIBUTION="None"
DESIRABLE_CLASS=4
LR=0.001
NUM_EPOCHS=120
TRAIN_SCRIPT="train.py"

# Process command line arguments
if [ $# -ge 1 ]; then
    DATA_DIR=$1
fi
if [ $# -ge 2 ]; then
    DATA_NAME=$2
fi
if [ $# -ge 3 ]; then
    PREV_WEIGHTS=$3
fi
if [ $# -ge 4 ]; then
    BATCH_SIZE=$4
fi
if [ $# -ge 5 ]; then
    CLASS_DISTRIBUTION=$5
fi
if [ $# -ge 6 ]; then
    DESIRABLE_CLASS=$6
fi
if [ $# -ge 7 ]; then
    LR=$7
fi
if [ $# -ge 8 ]; then
    NUM_EPOCHS=$8
fi
if [ $# -ge 9 ]; then
    TRAIN_SCRIPT=$9
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Create backup directory if it doesn't exist
BACKUP_DIR="./config_backups"
mkdir -p "$BACKUP_DIR"

# Create a backup with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/${CONFIG_FILE%.*}_${TIMESTAMP}.yaml"
cp "$CONFIG_FILE" "$BACKUP_FILE"
echo "Created backup of original config file at: $BACKUP_FILE"

# Update values in the config file using sed with specific patterns for nested YAML

# Update data directory
sed -i "s|^\(  dir: \).*|\1$DATA_DIR|" "$CONFIG_FILE"

# Update data name
sed -i "s|^\(  name: \).*|\1$DATA_NAME|" "$CONFIG_FILE"

# Update model previous weights
sed -i "s|^\(  prev_weights: \).*|\1$PREV_WEIGHTS|" "$CONFIG_FILE"

# Update batch size
sed -i "s|^\(  batch_size: \)[0-9]*|\1$BATCH_SIZE|" "$CONFIG_FILE"

# Update class distribution
sed -i "s|^\(  class_distribution: \).*|\1$CLASS_DISTRIBUTION|" "$CONFIG_FILE"

# Update desirable class
sed -i "s|^\(  desirable_class: \)[0-9]*|\1$DESIRABLE_CLASS|" "$CONFIG_FILE"

# Update learning rate
sed -i "s|^\(  lr: \)[0-9.]*|\1$LR|" "$CONFIG_FILE"

# Update number of epochs
sed -i "s|^\(  num_epochs: \)[0-9]*|\1$NUM_EPOCHS|" "$CONFIG_FILE"

echo "Updated config values:"
echo "  - data dir: $DATA_DIR"
echo "  - data name: $DATA_NAME"
echo "  - prev_weights: $PREV_WEIGHTS"
echo "  - batch_size: $BATCH_SIZE"
echo "  - class_distribution: $CLASS_DISTRIBUTION"
echo "  - desirable_class: $DESIRABLE_CLASS"
echo "  - lr: $LR"
echo "  - num_epochs: $NUM_EPOCHS"

# Check if train script exists
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script '$TRAIN_SCRIPT' not found!"
    exit 1
fi

# Check if config has actually changed from original
if ! cmp -s "$CONFIG_FILE" "$BACKUP_FILE"; then
    echo "Config file has been modified."
    
    # Create a timestamped version of the new config
    NEW_CONFIG="${BACKUP_DIR}/new_config_${TIMESTAMP}.yaml"
    cp "$CONFIG_FILE" "$NEW_CONFIG"
    echo "Saved new config to: $NEW_CONFIG"
else
    echo "No changes were made to the config file."
fi

# Run the training script
echo "Starting training..."
python "$TRAIN_SCRIPT" --config "$CONFIG_FILE"

echo "Training complete!"

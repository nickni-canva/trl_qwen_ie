#!/bin/bash

# Script to monitor GPU availability and run training when 8 GPUs are free
# Usage: bash monitor_and_run.sh

set -e

# Configuration
REQUIRED_GPUS=8
CHECK_INTERVAL=60  # Check every 60 seconds
MEMORY_THRESHOLD=500  # Consider GPU free if memory usage < 500 MB
UTILIZATION_THRESHOLD=10  # Consider GPU free if utilization < 10%

echo "=========================================="
echo "GPU Monitoring Script"
echo "=========================================="
echo "Required free GPUs: ${REQUIRED_GPUS}"
echo "Check interval: ${CHECK_INTERVAL} seconds"
echo "Memory threshold: ${MEMORY_THRESHOLD} MB"
echo "Utilization threshold: ${UTILIZATION_THRESHOLD}%"
echo "=========================================="
echo ""

# Function to count available GPUs
count_free_gpus() {
    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: nvidia-smi not found. Please ensure NVIDIA drivers are installed."
        exit 1
    fi
    
    # Count GPUs with low memory usage and low utilization
    # Using nvidia-smi to get memory used (MB) and GPU utilization (%)
    local free_count=0
    local total_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    
    # Get memory usage and utilization for all GPUs
    while IFS=',' read -r gpu_id memory_used utilization; do
        # Remove whitespace
        memory_used=$(echo "$memory_used" | tr -d ' ')
        utilization=$(echo "$utilization" | tr -d ' ')
        
        # Check if GPU is free (low memory and low utilization)
        if [ "$memory_used" -lt "$MEMORY_THRESHOLD" ] && [ "$utilization" -lt "$UTILIZATION_THRESHOLD" ]; then
            ((free_count++))
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    
    echo "$free_count"
}

# Function to display GPU status
display_gpu_status() {
    echo "Current GPU Status:"
    echo "-------------------"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory \
        --format=csv,noheader | while IFS=',' read -r idx name mem_used mem_total util_gpu util_mem; do
        echo "GPU $idx: $name | Memory: $mem_used/$mem_total | Util: $util_gpu (GPU) $util_mem (Mem)"
    done
    echo ""
}

# Main monitoring loop
echo "Starting GPU monitoring..."
echo ""

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    free_gpus=$(count_free_gpus)
    
    echo "[$timestamp] Free GPUs: $free_gpus / $REQUIRED_GPUS"
    
    if [ "$free_gpus" -ge "$REQUIRED_GPUS" ]; then
        echo ""
        echo "=========================================="
        echo "SUCCESS: $free_gpus GPUs are now available!"
        echo "=========================================="
        echo ""
        display_gpu_status
        echo ""
        echo "Starting training script..."
        echo "Command: cd ~/work/trl_qwen_image_edit && bash run_8gpu.sh"
        echo ""
        
        # Change directory and run the training script
        cd ~/work/trl_qwen_image_edit
        bash run_8gpu.sh
        
        echo ""
        echo "=========================================="
        echo "Training script completed!"
        echo "=========================================="
        exit 0
    else
        # Show detailed status every 5 checks (5 minutes if CHECK_INTERVAL=60)
        if [ $(($(date +%s) % 300)) -lt "$CHECK_INTERVAL" ]; then
            display_gpu_status
        fi
    fi
    
    # Wait before next check
    sleep "$CHECK_INTERVAL"
done











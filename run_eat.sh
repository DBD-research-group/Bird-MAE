#!/bin/bash

# define the base paths
PATHS=(
    "/home/lrauch/mnt_check/pretrain_xcl_eat_base/runs/XCL/EAT/2025-07-27_125911/callback_checkpoints"
    "/home/lrauch/mnt_check/pretrain_xcl_eat_base/runs/XCL/EAT/2025-07-27_105702/callback_checkpoints"
)

# define epochs with zero-padding (04, 09, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59)
EPOCHS=(04 09 14 19 24 29 34 39 44 49 54 59)

# extract run identifier from path for naming
get_run_id() {
    local path=$1
    echo $(basename $(dirname $path))
}

echo "Starting EAT checkpoint experiments..."
echo "Experiment config: $EXPERIMENT_CONFIG"
echo "=========================================="

total_experiments=$((${#PATHS[@]} * ${#EPOCHS[@]}))
current_experiment=0

for path in "${PATHS[@]}"; do
    run_id=$(get_run_id "$path")
    echo ""
    echo "Processing run: $run_id"
    echo "Path: $path"
    
    for epoch in "${EPOCHS[@]}"; do
        current_experiment=$((current_experiment + 1))
        # Use the zero-padded epoch number
        checkpoint_file="$path/EAT_XCL_epoch=${epoch}.ckpt"
        
        echo ""
        echo "[$current_experiment/$total_experiments] Running experiment:"
        echo "  Run ID: $run_id"
        echo "  Epoch: $epoch"
        echo "  Checkpoint: $checkpoint_file"
        
        # Check if checkpoint file exists
        if [ ! -f "$checkpoint_file" ]; then
            echo "  WARNING: Checkpoint file not found, skipping: $checkpoint_file"
            continue
        fi
        
        # Create unique experiment name (use epoch without leading zero for naming)
        epoch_no_zero=$((10#$epoch))  # Convert to decimal to remove leading zero
        echo "  Starting training..."
        
        # Run the experiment
        python finetune.py  \
            experiment=finetune_examples/finetune_hsn_eat.yaml \
            module.network.pretrained_weights_path=${checkpoint_file//=/\\=} \
        
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "  ✓ SUCCESS: Experiment completed successfully"
        else
            echo "  ✗ FAILED: Experiment failed with exit code $exit_code"
            # Optional: uncomment to stop on first failure
            # exit $exit_code
        fi
        
        echo "  ----------------------------------------"
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Total experiments run: $current_experiment"
#!/bin/bash

echo "======================================================"
echo "[$(date)] Quick FNPE Test - Verifying pipeline works"
echo "======================================================"

# Minimal settings for fast testing
NUM_SIM=100              # Very few simulations
T_OBS=50                 # Short trajectories
EPOCHS=2                 # Just 2 epochs
STEPS_PER_EPOCH=100      # Very few steps per epoch
BATCH_SIZE=32            # Small batch
HIDDEN_DIM=32            # Small network
NUM_HIDDEN=2             # Shallow network
NUM_DIFF_STEPS=50        # Few diffusion steps
NUM_POST_SAMPLES=50      # Few posterior samples
SEED=42

# Change to code directory (adjust path as needed)
cd "$(dirname "$0")/.." || exit 1

echo ""
echo "[CONFIG] Quick test settings:"
echo "  num_sim=$NUM_SIM, T_obs=$T_OBS, epochs=$EPOCHS"
echo "  steps_per_epoch=$STEPS_PER_EPOCH, batch_size=$BATCH_SIZE"
echo "  hidden_dim=$HIDDEN_DIM, num_hidden=$NUM_HIDDEN"
echo ""

# Run the experiment
python -m inference.fnpe_experiment \
    --exp-name "fnpe_quick_test" \
    --num-sim ${NUM_SIM} \
    --T-obs ${T_OBS} \
    --epochs ${EPOCHS} \
    --steps-per-epoch ${STEPS_PER_EPOCH} \
    --batch-size ${BATCH_SIZE} \
    --hidden-dim ${HIDDEN_DIM} \
    --num-hidden ${NUM_HIDDEN} \
    --model-type gru \
    --lr 5e-4 \
    --num-diff-steps ${NUM_DIFF_STEPS} \
    --num-post-samples ${NUM_POST_SAMPLES} \
    --seed ${SEED} \
    --params "mu,cd,m"

EXIT_CODE=$?

echo ""
echo "======================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] SUCCESS - FNPE pipeline test passed!"
else
    echo "[$(date)] FAILED - Exit code: $EXIT_CODE"
fi
echo "======================================================"

exit $EXIT_CODE

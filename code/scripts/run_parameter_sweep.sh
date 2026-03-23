#!/bin/bash -l
# ==============================================================================
# Parameter Sweep: Test different wheel radius and torque/brake scaling
#
# 6 configurations:
# 0: Baseline (radius_tire=0.3116, no scaling)
# 1: Larger wheel (+10%): radius_tire=0.343
# 2: Larger wheel (+20%): radius_tire=0.374
# 3: Smaller wheel (-10%): radius_tire=0.280
# 4: Motor torque scaled 1.2x
# 5: Brake pressure scaled 1.5x
#
# Usage: sbatch scripts/run_parameter_sweep.sh
# ==============================================================================

#SBATCH --job-name=param_sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=param_sweep_%j.out
#SBATCH --error=param_sweep_%j.err

set -e

echo "=================================================="
echo "[$(date)] Parameter Sweep Experiments"
echo "=================================================="

# --- Go to code directory ---
cd /bigwork/nhkbarit/thesis-code/code

# --- Load environment ---
module load Miniforge3
conda activate /software/NHKB22930/nhkbarit/conda_envs/npe

# --- JAX configuration ---
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORM_NAME || true

# --- Common args ---
COMMON_ARGS="--num-sim 5000 --T-seg 3000 --num-epochs 50 --stop-after-epochs 15 --device cuda --method npe --no-sbc --no-swd --no-one-step"
REAL_DATA="--real-data-csv ../data/measurements/Jeversen_2022_10_12_110132.csv"

# Backup original VehicleModel.py
cp simulation/VehicleModel.py simulation/VehicleModel.py.backup

# ==============================================================================
# Configuration 0: Baseline
# ==============================================================================
echo ""
echo ">>> [1/6] Baseline configuration..."
echo "    radius_tire=0.3116, torque_scale=1.0, brake_scale=1.0"
echo ""

# Reset to original
cp simulation/VehicleModel.py.backup simulation/VehicleModel.py

srun python run.py --exp-name sweep_baseline $COMMON_ARGS $REAL_DATA

# ==============================================================================
# Configuration 1: Larger wheel (+10%)
# ==============================================================================
echo ""
echo ">>> [2/6] Larger wheel radius (+10%)..."
echo "    radius_tire=0.343"
echo ""

cp simulation/VehicleModel.py.backup simulation/VehicleModel.py
sed -i 's/radius_tire = 0.3116/radius_tire = 0.343/' simulation/VehicleModel.py

srun python run.py --exp-name sweep_radius_plus10 $COMMON_ARGS $REAL_DATA

# ==============================================================================
# Configuration 2: Larger wheel (+20%)
# ==============================================================================
echo ""
echo ">>> [3/6] Larger wheel radius (+20%)..."
echo "    radius_tire=0.374"
echo ""

cp simulation/VehicleModel.py.backup simulation/VehicleModel.py
sed -i 's/radius_tire = 0.3116/radius_tire = 0.374/' simulation/VehicleModel.py

srun python run.py --exp-name sweep_radius_plus20 $COMMON_ARGS $REAL_DATA

# ==============================================================================
# Configuration 3: Smaller wheel (-10%)
# ==============================================================================
echo ""
echo ">>> [4/6] Smaller wheel radius (-10%)..."
echo "    radius_tire=0.280"
echo ""

cp simulation/VehicleModel.py.backup simulation/VehicleModel.py
sed -i 's/radius_tire = 0.3116/radius_tire = 0.280/' simulation/VehicleModel.py

srun python run.py --exp-name sweep_radius_minus10 $COMMON_ARGS $REAL_DATA

# ==============================================================================
# Configuration 4: Motor torque scaled 1.2x
# ==============================================================================
echo ""
echo ">>> [5/6] Motor torque scaled 1.2x..."
echo ""

cp simulation/VehicleModel.py.backup simulation/VehicleModel.py
# Scale engine torque by 1.2 in the vehicle_dx function
sed -i 's/engine_torque = aux_input\["engine_torque"\]/engine_torque = aux_input["engine_torque"] * 1.2/' simulation/VehicleModel.py

srun python run.py --exp-name sweep_torque_1p2x $COMMON_ARGS $REAL_DATA

# ==============================================================================
# Configuration 5: Brake pressure scaled 1.5x
# ==============================================================================
echo ""
echo ">>> [6/6] Brake pressure scaled 1.5x..."
echo ""

cp simulation/VehicleModel.py.backup simulation/VehicleModel.py
# Scale brake torque by 1.5 in the vehicle_dx function
sed -i 's/break_torque = aux_input\["break_torque"\]/break_torque = aux_input["break_torque"] * 1.5/' simulation/VehicleModel.py

srun python run.py --exp-name sweep_brake_1p5x $COMMON_ARGS $REAL_DATA

# ==============================================================================
# Restore original
# ==============================================================================
echo ""
echo ">>> Restoring original VehicleModel.py..."
cp simulation/VehicleModel.py.backup simulation/VehicleModel.py
rm simulation/VehicleModel.py.backup

echo ""
echo "=================================================="
echo "[$(date)] Parameter sweep complete!"
echo "Results in: experiments/sweep_*/"
echo "=================================================="
echo ""
echo "Compare results:"
echo "  - sweep_baseline_*"
echo "  - sweep_radius_plus10_*"
echo "  - sweep_radius_plus20_*"
echo "  - sweep_radius_minus10_*"
echo "  - sweep_torque_1p2x_*"
echo "  - sweep_brake_1p5x_*"
echo "=================================================="

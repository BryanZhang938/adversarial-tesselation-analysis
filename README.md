# ReLU Tessellation Dynamics: Standard vs Adversarial Training

Experimental pipeline for MATH 216B final project investigating how optimization
shapes ReLU network tessellations under standard and adversarial training.

## Setup

### 1. Create conda environment
```bash
conda create -n tessellation python=3.10
conda activate tessellation
```

### 2. Install core dependencies
```bash
pip install torch torchvision matplotlib numpy scipy seaborn tqdm pandas
```

### 3. Install SplineCam (for tessellation computation)
SplineCam requires `graph-tool`, which is easiest to install via conda:
```bash
conda install -c conda-forge graph-tool
pip install python-igraph>=0.10 networkx
git clone https://github.com/AhmedImtiazPrio/splinecam.git
cd splinecam && pip install -e .
```

If `graph-tool` is difficult on your system, see the Colab demo:
https://bit.ly/splinecam-demo

## Project Structure
```
tessellation_project/
├── configs/
│   └── experiment_config.py      # Hyperparameters and experiment settings
├── src/
│   ├── datasets.py               # Synthetic 2D dataset generators
│   ├── models.py                 # ReLU MLP architectures
│   ├── train.py                  # Standard and adversarial training loops
│   ├── adversarial.py            # PGD attack implementation
│   ├── tessellation_analysis.py  # SplineCam wrapper + statistics computation
│   └── visualization.py          # Plotting functions for paper figures
├── notebooks/
├── figures/
├── checkpoints/
├── run_experiment.py             # Main experiment runner
└── README.md
```

## Running Experiments
```bash
python run_experiment.py
```
This will:
1. Generate synthetic datasets (spirals, concentric rings)
2. Train networks under standard ERM and PGD-AT
3. Save checkpoints at regular intervals
4. Compute tessellation statistics at each checkpoint
5. Generate comparison plots in `figures/`

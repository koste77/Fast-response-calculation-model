# BEP-PINO

[简体中文](README.md) | **English**

BEP-PINO is a research project on physics-informed neural operators for predicting Euler–Bernoulli beam responses. The repository provides scripts for model training, evaluation, finite element method (FEM) comparison, model benchmarking, and ablation studies under various boundary and loading conditions.

The models take the beam load distribution and physical parameters as inputs and predict the following response quantities:

- Deflection $u$
- Rotation $\phi$
- Bending moment $M$
- Shear force $Q$

## Project Contents

### Multi-scenario BEP-PINO

`BEP-PINO/BEP-PINO Model comparison under multiple operating conditions/` contains the following scenarios:

- Simply supported beams under linear, quadratic, and concentrated loads
- Fixed-fixed beams under linear, quadratic, and concentrated loads
- Cantilever beams under linear, quadratic, and concentrated loads
- Fixed-fixed beams with left-end rotation or settlement
- Multi-scenario Latin hypercube sampling (LHS) validation and result visualization

Each scenario directory generally contains:

- `BEP-PINO_Model_*.py`: model definition and training
- `BEP-PINO_Model_*_EVA.py`: loading trained results for evaluation, error analysis, and visualization

### Model Comparison and Ablation Studies

`BEP-PINO/Model comparison/` contains:

- BEP-PFNO
- HO-BEP-PINO
- ME-BEP-PINO
- BEP-PDON
- P-ml-PINN
- SC-ml-PINN
- Multi-model comparison plots
- Model ablation and parameter sensitivity experiments

## Repository Structure

```text
.
├── README.md
├── README_EN.md
└── BEP-PINO/
    ├── BEP-PINO Model comparison under multiple operating conditions/
    │   ├── Cantilever/
    │   ├── Fixed-fixed/
    │   ├── Fixed-fixed left rotation/
    │   ├── Fixed-fixed left settlement/
    │   ├── simply supported/
    │   └── Multi-condition verification/
    └── Model comparison/
        ├── 0 Model comparison drawing/
        ├── 1 Model ablation/
        ├── BEP-PDON/
        ├── BEP-PFNO/
        ├── HO-BEP-PINO/
        ├── ME-BEP-PINO/
        ├── P-ml-PINN/
        └── SC-ml-PINN/
```

## Requirements

- Python 3.10 or later (recommended)
- PyTorch
- NumPy
- SciPy
- Matplotlib
- Pandas
- TensorFlow (required only by the P-ml-PINN and SC-ml-PINN scripts)

Using an isolated virtual environment is recommended:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy scipy matplotlib pandas torch tensorflow
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy scipy matplotlib pandas torch tensorflow
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/koste77/git.git
cd git
```

### 2. Train a Single Scenario

The following command uses a fixed-fixed beam under a concentrated load as an example:

```bash
python "BEP-PINO/BEP-PINO Model comparison under multiple operating conditions/Fixed-fixed/Fixed-fixed concentrated load/BEP-PINO_Model_fixed-fixed_concentrated.py"
```

The training script saves the model checkpoint and training logs in the corresponding scenario directory. The default number of training epochs is relatively large, so the runtime depends on the available hardware. The number of training epochs, sample count, and optimizer parameters can be adjusted in the main configuration section of the script.

### 3. Evaluate the Trained Model

After training, run the corresponding evaluation script:

```bash
python "BEP-PINO/BEP-PINO Model comparison under multiple operating conditions/Fixed-fixed/Fixed-fixed concentrated load/BEP-PINO_Model_fixed-fixed_concentrated_EVA.py"
```

The evaluation script loads the model checkpoint from the same directory and generates response curves, error metrics, or related figures.

### 4. Run Multi-scenario Validation

First train all required scenarios and confirm that their model and log files exist. Then run:

```bash
python "BEP-PINO/BEP-PINO Model comparison under multiple operating conditions/Multi-condition verification/run_multiscenario_lhs100.py" --samples 100 --device cpu
```

Results are saved to the `outputs/` directory beside the script by default.

Generate summary figures with:

```bash
python "BEP-PINO/BEP-PINO Model comparison under multiple operating conditions/Multi-condition verification/plot_multiscenario_results.py"
```

### 5. Run Ablation Studies

Run the ablation workflow in the order training → evaluation → plotting:

```bash
cd "BEP-PINO/Model comparison/1 Model ablation"
python "BP-PFNO_3.3_Ablation_and_Param_Train.py"
python "BP-PFNO_3.3_Ablation_and_Param_EVA.py"
python "BP-PFNO_3.3_Ablation_and_Param_Plot.py"
```

For an initial test, set `RUN_MODE` to `"smoke"` near the top of the training script. This checks the workflow with fewer Adam epochs and L-BFGS iterations.

## Output Files

Different scripts generate one or more of the following file types:

- `*.pth`: PyTorch model checkpoints or training logs
- `*.npz`: training or evaluation data in NumPy format
- `*.csv`: per-scenario metrics and summary results
- `*.json`: experiment configurations, metadata, or representative response curves
- `*.png`, `*.pdf`, `*.svg`: result figures and publication graphics

> Note: This repository currently contains mainly source code. Evaluation scripts generally depend on model weights and logs generated during training. If an evaluation script reports a missing file, run the corresponding training script first.

## Usage Notes

- Each script contains its own model parameters and training configuration. Review the settings near the top of the file and around `if __name__ == "__main__"` before running it.
- Because directory and file paths contain spaces, enclose complete paths in double quotation marks when running commands.
- Training uses double-precision floating-point arithmetic, so full experiments may require substantial time and memory.
- Most training scripts set NumPy and PyTorch random seeds to improve reproducibility.

## License and Citation

This repository does not currently include an open-source license. Before reusing or distributing the code, or using it in public work, contact the project maintainer to confirm the applicable permissions.

If this project contributes to your research, consider adding a BibTeX entry here when the related paper or formal citation information becomes available.

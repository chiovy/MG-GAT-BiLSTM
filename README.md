# MG-GAT-BiLSTM Anomaly Detection Repository

This repository provides the implementation of the MG-GAT-BiLSTM model for multivariate time-series anomaly detection, together with ablation experiments, threshold analysis, and visualization scripts. It mainly targets UAV sensor/flight data, but the code is applicable to general multivariate time series.

The repository includes:

- MG-GAT-BiLSTM and several model variants
- Comparisons with baseline models: MLP, BiLSTM, CNN-LSTM, GRU, Transformer
- Ablation experiments on the global attention and multi-graph modules
- Multiple unsupervised thresholding methods (POT-EVT, MAD, IQR, 3-Sigma)
- Radar charts and anomaly score visualizations

## Project Structure

```text
MG-GAT-BiLSTM/                      # Root directory
├─ DataSet/                         # Dataset description
│  └─ Readme.md                     # Links to ALFA / UAV GPSAttack datasets
├─ Model/                           # Model definitions
│  ├─ MG_GAT_BiLSTM.py              # Main model: multi-graph GAT + BiLSTM + global attention + EVT
│  ├─ MG_GAT_BiLSTM_NoAttention.py  # Ablation: without global attention
│  ├─ MG_GAT_BiLSTM_NoMIC.py        # Ablation: without MIC graph
│  ├─ MG_GAT_BiLSTM_NoDCOR.py       # Ablation: without dCor graph
│  ├─ MG_GAT_BiLSTM_NoSpearman.py   # Ablation: without Spearman graph
│  ├─ MLP.py                        # Baseline: MLP
│  ├─ BiLSTM.py                     # Baseline: BiLSTM
│  ├─ CNNLSTM.py                    # Baseline: CNN-LSTM
│  ├─ GRU.py                        # Baseline: GRU
│  └─ Transformer.py                # Baseline: Transformer
├─ Rada/                            # Radar-chart visualizations
│  ├─ ALFA.py                       # Radar chart for ALFA dataset (English)
│  └─ GPSattack.py                  # Radar chart for UAV GPSAttack scenario
├─ anomalyscorevis/                 # Anomaly score visualizations
│  ├─ MG-GAT-BiLSTM.png
│  ├─ BiLSTM.png
│  ├─ CNN-LSTM.png
│  ├─ GRU.png
│  ├─ MLP.png
│  ├─ Transformer.png
│  ├─ Without MIC.png
│  ├─ Without Spearman.png
│  ├─ Without dCor.png
│  └─ Without global attention.png
├─ training/                        # Generic training and strict evaluation entry points (wrap upstream framework)
│  ├─ train.py                      # Generic training entry (wrapper around parent train.py)
│  └─ evaluate.py                   # Strict evaluation entry (wrapper around parent evaluate.py)
├─ util/                            # Utility functions
│  ├─ mic_utils.py                  # MIC / mutual-information-based feature selection
│  ├─ normalization_utils.py        # Normalization and preprocessing utilities
│  └─ anomaly_visualization_utils.py# Single-feature anomaly vs normal visualization
├─ AblationExperiment.py            # Aggregation and visualization of ablation-study results
├─ ThresholdCompare.py              # Comparison of different thresholding methods
└─ README.md                        # This file
```

## Requirements

The core code is implemented with PyTorch and NumPy/Scikit-Learn. Some components depend on `torch_geometric`, and `minepy` is optionally used for MIC computation. Recommended Python version is 3.9–3.11.

- Python 3.9+
- torch (version matching your CUDA setup is recommended)
- torch-geometric (for GATv2Conv-based graph convolutions in the main model)
- numpy, pandas
- scipy
- scikit-learn
- matplotlib
- minepy (optional, for accurate MIC calculation; falls back to mutual information if missing)

You can install the common dependencies with (adjust `torch` and `torch-geometric` according to your CUDA version):

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib minepy
```

For `torch_geometric`, please follow the official installation instructions, for example:

```bash
pip install torch-geometric -f https://data.pyg.org/whl/torch-$(python -c "import torch;print(torch.__version__)").html
```

> Note: `training/train.py` and `training/evaluate.py` load `train.py` and `evaluate.py` from the parent directory using relative paths. This is designed to integrate with upstream benchmarks such as ALFA. Before running, ensure this repository is placed under a parent directory that contains those scripts.

## Data Preparation

This repository is designed to run on the ALFA UAV anomaly detection benchmark and the UAV GPSAttack dataset. Dataset links are listed in `DataSet/Readme.md`:

- ALFA dataset: https://theairlab.org/alfa-dataset/
- UAV GPSAttack dataset: https://ieee-dataport.org/open-access/uav-attack-dataset

The general data preparation pipeline is:

1. Download the raw datasets and preprocess them according to your upstream benchmark framework, producing train/validation/test splits and corresponding anomaly labels.
2. Construct MIC, dCor, and Spearman correlation graphs as needed (you can use `util/mic_utils.py` together with your own preprocessing scripts).
3. Place the preprocessed sequences and graph structures in the directories expected by the upstream training framework so that `training/train.py` and `training/evaluate.py` can load them correctly.

Because the training entry points in this repository are wrappers around the parent directory `train.py`/`evaluate.py`, please refer to the documentation of your upstream framework for the exact data file naming and directory structure.

## Quick Start

The following steps assume you already have an upstream training framework and preprocessed data. They show how to use this repository for training, ablation studies, and threshold analysis.

1. Train the MG-GAT-BiLSTM main model

   ```bash
   cd MG-GAT-BiLSTM

   python training/train.py \
     --data_dir <preprocessed_data_dir> \
     --config <optional: JSON config file> \
     --batch_size 64 \
     --epochs 100 \
     --learning_rate 1e-3 \
     --output_dir outputs/MG_GAT_BiLSTM
   ```

   After training, the best model and training summary will be saved under the specified `output_dir`.

2. Strict evaluation and anomaly score export

   ```bash
   python training/evaluate.py \
     --model_path <path_to_best_model.pth> \
     --data_dir <test_sequences_dir> \
     --output_dir Images/MG_GAT_BiLSTM \
     --mask_length 150 \
     --precision_mode normal \
     --recalibrate \
     --recalibrate_method pot \
     --auto_adjust_threshold
   ```

   This script calls the evaluation module in the parent directory, performs strict evaluation, and saves anomaly scores, thresholds, and visualizations to the specified directory.

3. Summarize ablation study results

   After training and evaluating the variants without attention/MIC/dCor/Spearman, run:

   ```bash
   python AblationExperiment.py
   ```

   The script searches default result directories for `evaluation_summary.json` from each ablation experiment, aggregates metrics such as Precision/Recall/F1/ROC-AUC, and generates comparison tables and bar charts.

4. Compare thresholding methods

   When you have anomaly scores on the validation set (single `.npy` file) and anomaly scores plus labels for multiple test flights, run:

   ```bash
   python ThresholdCompare.py \
     --val_scores path/to/val_scores.npy \
     --test_scores_dir path/to/test_scores_dir \
     --output_dir Images/ThresholdCompare
   ```

   The `test_scores_dir` should contain files named as:

   - `scores_flight_XXX.npy`
   - `labels_flight_XXX.npy`

   The script compares POT-EVT, MAD, IQR, and 3-Sigma thresholding methods over multiple flights, reporting and visualizing Precision/Recall/F1/ROC-AUC.

5. Baseline comparison and visualization

   - Radar charts: use `Rada/ALFA.py` and `Rada/GPSattack.py` to generate radar charts comparing MG-GAT-BiLSTM against baseline models on the ALFA and UAV GPSAttack datasets.
   - Anomaly score plots: PNG images of anomaly scores for the main model, ablation variants, and baselines are stored in `anomalyscorevis/`.

## Main Scripts

- `Model/MG_GAT_BiLSTM.py`: Main multi-graph GAT-BiLSTM model with MIC/dCor/Spearman branches, global attention, BiLSTM temporal modeling, and a POT-EVT thresholding function for unsupervised threshold selection.
- `Model/MG_GAT_BiLSTM_NoAttention.py`, `Model/MG_GAT_BiLSTM_NoMIC.py`, `Model/MG_GAT_BiLSTM_NoDCOR.py`, `Model/MG_GAT_BiLSTM_NoSpearman.py`: Ablation variants removing global attention or specific graph branches to analyze their contributions.
- `training/train.py`: Generic training entry point that wraps the parent directory `train.py`, allowing command-line overrides of batch size, number of epochs, and learning rate.
- `training/evaluate.py`: Strict evaluation entry point wrapping the parent directory `evaluate.py`, supporting various threshold recalibration methods and strictness levels.
- `AblationExperiment.py`: Aggregates flight-level metrics from each ablation experiment’s `evaluation_summary.json`, writes CSV summaries, and generates bar plots for use in papers.
- `ThresholdCompare.py`: Compares multiple unsupervised thresholding methods on fixed anomaly scores, producing metric tables and visualizations.
- `util/mic_utils.py`: MIC or mutual-information-based feature-selection utilities, useful for constructing MIC graphs and preprocessing.
- `util/anomaly_visualization_utils.py`: Visualizes normal vs anomalous segments for a single feature, useful for examples and debugging.

## Contact

If you have any questions or suggestions, please submit an issue or contact the author by email: wubo@nchu.edu.cn.

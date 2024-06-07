<p align=center>
    <img src="gfx/gpn-extensions-dark.svg#gh-dark-mode-only" alt="GPN Extensions">
    <img src="gfx/gpn-extensions.svg#gh-light-mode-only" alt="GPN Extensions">
</p>
<h1 align=center style="clear:both;">LOP-GPN & CUQ-GNN</h1>
<p align=center><b>Extensions of Graph Posterior Networks</b></p>
<hr style="clear:both;">

This is the official implementation of the [*Linear Opinion Pooled Graph Posterior Network* (**LOP-GPN**)](https://openreview.net/forum?id=qLGkfpXTSn) and *Committee-based Graph Uncertainty Quantification using Posterior Networks* (**CUQ-GNN**) models by Damke and Hüllermeier.
The implementation is based on the official implementation of [Graph Posterior Networks](https://github.com/stadlmax/Graph-Posterior-Network) by Stadler et al. (see [citation](#Cite)).

The following additions were made:
- Implementation of LOP-GPN
- Implementation of CUQ-GNN
- Support for additional uncertainty measures
- Accuracy-rejection curve generation
- Added the `eval.clj` script for automatic evaluation and summarization of the experiments
- Migrated the conda environment to Poetry
- Added and corrected type annotations

## Installation

The installation requires [Poetry](https://python-poetry.org/).
All dependencies can then be installed in a virtual environment simply by running:
```sh
poetry install
```

The evaluation and result aggregation script ([`eval.clj`](./eval.clj)) requires the [Babashka](https://babashka.org/) scripting runtime.

## I just want to reproduce the experiments!

To replicate the results from the paper, you can simply run the following commands:
```sh
./eval.clj eval # Train and evaluate all model variants
./eval.clj acc-rej-tables # Aggregate the evaluation results into accuracy-rejection curves (CSV)
./eval.clj id-ood-table # Aggregate the OOD evaluation results into a table (CSV)
```
The trained model weights are written to `saved_experiments` and their aggregated evaluation results are written to `results`.
Accuracy-rejection curves and the OOD evaluation table are written as `*.csv` files to a `tables` directory.

Note that the implementation has only been tested on Linux; it might not run on MacOS or Windows.

### Running only parts of the evaluation

By default, the `eval` command will evaluate all combinations of models and datasets in different settings (standard classification, OOD with left-out-classes, OOD with Gaussian noise or OOD with Bernoulli noise).
This behavior can be configured by the following command line options:

- `-d`, `--dataset S`  : Datasets (`CoraML`, `CiteSeerFull`, `AmazonPhotos`, `AmazonComputers`, `PubMedFull`, `ogbn-arxiv`)
- `-m`, `--model S`    : Models (`appnp`, `matern_ggp`, `gdk`, `gpn`, `gpn_rw`, `gpn_lop`, `cuq_appnp`, `cuq_gcn`, `cuq_gat`)
- `-s`, `--setting S`  : Settings (`classification`, `ood_loc`, `ood_features_normal`, `ood_features_ber`)
- `-o`, `--override k=v` : Override key, value pairs (can be used to reconfigure/override arbitrary run parameters)
- `--dry`         : Dry run (only print which evaluations would be performed).
- `--retrain`     : Models will be trained, even if model parameters are already stored for a given run (in `saved_experiments`).
- `--reeval`      : Models will be evaluated, even if evaluation results are already stored for a given run (in `saved_experiments`).
- `--no-cache`    : Runs will be considered, even if there are already cached results (in `results`).
- `--delete`      : All results for the selected runs are deleted from `save_experiments` and `results`.

The first four options (`-d`, `-m`, `-s`, `-o`) also accept multiple values, which can be provided by adding the options multiple times.

**Example:**
```shell
./eval.clj eval -d CoraML -d AmazonComputers -m gpn -m gpn_lop -m cuq_gat -s classification -o run.num_splits=2
```
This command will evaluate the GPN, LOP-GPN and GAT-based CUQ-GNN models on the CoraML and AmazonComputers datasets in the standard classification setting (no OOD transformation of the training data).
Additionally, it will only run the first two (of 10) train/test splits and only compute the average of the results over those first two splits.

Generally, overrides are configured via `key=value` pairs.
A key is of the form `config_type.param_name`; `config_type` is either `run`, `data`, `model` or `training`; given a `config_type`, the valid `param_name`s are defined [here](./gpn/utils/config.py).

## Structure of the project

As mentioned above, this implementation is directly based on the [reference implementation](https://github.com/stadlmax/Graph-Posterior-Network) of GPNs by Stadler et al.
The overall structure is documented in the README of their repository.

The main implementation of LOP-GPN is split across the following files:
- [`gpn_lop.py`](./gpn/models/gpn_lop.py): Contains the core implementation of LOP-GPN, based on `gpn_base.py`.
- [`gpn_base.py`](./gpn/models/gpn_base.py): The GPN base implementation was made more generic to make it compatible with LOP-GPN.
- [`loss.py`](./gpn/nn/loss.py#L58): The loss for LOP-GPN is computed via the `mixture_uce_loss` and `categorical_entropy_reg` functions.
- [`appnp_propagation.py`](./gpn/layers/appnp_propagation.py): The APPNP implementation was extended by support for fully-sparse feature and adjacency matrices, to make LOP-GPN computationally tractable.

The implementation of CUQ-GNN is split across the following files:
- [`cuq_gnn.py`](./gpn/models/cuq_gnn.py): Contains the cor implementation of CUQ-GNN, based on `gpn_base.py`.
- [`gpn_base.py`](./gpn/models/gpn_base.py): The GPN base implementation was made more generic to make it compatible with CUQ-GNN.

## Cite

If you use LOP-GPN, CUQ-GNN or this code in your own work, please cite, both, our paper(s) and that by Stadler et al.

### [LOP-GPN](https://openreview.net/forum?id=qLGkfpXTSn)
```
@inproceedings{lop-gpn,
title={Linear Opinion Pooling for Uncertainty Quantification on Graphs},
author={Damke, Clemens and H{\"u}llermeier, Eyke},
booktitle={The 40th Conference on Uncertainty in Artificial Intelligence},
year={2024},
url={https://openreview.net/forum?id=qLGkfpXTSn}
}
```

### CUQ-GNN
The reference to our CUQ-GNN paper will be added later.


### GPN
```
@incollection{graph-postnet,
title={Graph Posterior Network: Bayesian Predictive Uncertainty for Node Classification},
author={Stadler, Maximilian and Charpentier, Bertrand and Geisler, Simon and Z{\"u}gner, Daniel and G{\"u}nnemann, Stephan},
booktitle = {Advances in Neural Information Processing Systems},
volume = {34},
publisher = {Curran Associates, Inc.},
year = {2021}
}
```

# CUQ-GNN: Committee-based Graph Uncertainty Quantification using Posterior Networks

<center><img src="cuq-gnn.png" alt="CUQ-GNN"></center>

This is the implementation of the CUQ-GNN model, it is based on the official implementation of [Graph Posterior Networks](https://github.com/stadlmax/Graph-Posterior-Network) by Maximilian Stadler.

The following additions were made:
- Implementation of CUQ-GNN
- Support for additional uncertainty measures
- Accuracy-rejection curve generation
- Added the `eval.clj` script for automatic evaluation and summarization of the experiments.
- Migrated the conda environment to Poetry.

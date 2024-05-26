# **GraphSpec: An Advanced Model for Peptide Spectrum Prediction**
![output-onlinepngtools](https://github.com/commashy/GraphSpec/assets/96168749/6452b892-a265-4002-b9a3-b5d7050cc8f5)
This repository contains the code and resources for GraphSpec, a novel model designed for peptide spectrum prediction using advanced graph neural network (GNN) technologies. Developed as part of a final year thesis, GraphSpec aims to address the limitations of traditional MS/MS data interpretation and enhance the accuracy of peptide identification through the integration of molecular graph embeddings and deep learning techniques.

## **Overview**
![output-onlinepngtools (1)](https://github.com/commashy/GraphSpec/assets/96168749/6525c5a6-22c0-41c1-b303-b3f52e0482db)
GraphSpec features two primary components:
1. **Graph Embedder**: Utilizes Graph Attention Network v2 (GATv2) and Transformer Convolution Layer (TransformerConv) to convert peptide sequences into detailed molecular graphs.
![output-onlinepngtools (2)](https://github.com/commashy/GraphSpec/assets/96168749/c3dbaadb-d83c-4831-aad9-f8d86a508b3a)
2. **Spectral Prediction Model**: Employs ResNet and ConvNeXt architectures to predict tandem mass spectra from the generated molecular graph embeddings.
![output-onlinepngtools (3)](https://github.com/commashy/GraphSpec/assets/96168749/92bc2d62-5afe-4ea6-bcb1-f30c59f72b02)

## **Key Features**

- **State-of-the-Art Performance**: Demonstrates superior accuracy in predicting spectra for standard peptides compared to existing models like Seq2MS.
- **PTM Handling**: Shows potential for improved handling of post-translational modifications (PTMs) with fine-tuning strategies.
- **Comprehensive Training**: Trained on extensive datasets including ProteomeTools HCD, NIST Human HCD, and NIST Synthetic HCD.
- **Fine-Tuning Strategies**: Implements strategies focusing on training set size and diversity to enhance performance, particularly for PTMs.

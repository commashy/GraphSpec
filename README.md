# **GraphSpec: An Advanced Model for Peptide Spectrum Prediction**

This repository contains the code and resources for GraphSpec, a novel model designed for peptide spectrum prediction using advanced graph neural network (GNN) technologies. Developed as part of a final year thesis, GraphSpec aims to address the limitations of traditional MS/MS data interpretation and enhance the accuracy of peptide identification through the integration of molecular graph embeddings and deep learning techniques.

## **Overview**

GraphSpec features two primary components:

1. **Graph Embedder**: Utilizes Graph Attention Network v2 (GATv2) and Transformer Convolution Layer (TransformerConv) to convert peptide sequences into detailed molecular graphs.
2. **Spectral Prediction Model**: Employs ResNet and ConvNeXt architectures to predict tandem mass spectra from the generated molecular graph embeddings.

## **Key Features**

- **State-of-the-Art Performance**: Demonstrates superior accuracy in predicting spectra for standard peptides compared to existing models like Seq2MS.
- **PTM Handling**: Shows potential for improved handling of post-translational modifications (PTMs) with fine-tuning strategies.
- **Comprehensive Training**: Trained on extensive datasets including ProteomeTools HCD, NIST Human HCD, and NIST Synthetic HCD.
- **Fine-Tuning Strategies**: Implements strategies focusing on training set size and diversity to enhance performance, particularly for PTMs.

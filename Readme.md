# Transformer Network Implementation in PyTorch

## Objective
Implement a transformer network in Python using the PyTorch framework for multi-class classification of environmental audio recordings categorized into 10 different classes.

## Dataset
The dataset consists of 400 environmental audio recordings categorized into 10 different classes.[Data Repository](https://iitjacin-my.sharepoint.com/personal/mishra_10_iitj_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmishra%5F10%5Fiitj%5Fac%5Fin%2FDocuments%2FAudio%5FAssignment%5FDL%2FArchive%2Ezip&parent=%2Fpersonal%2Fmishra%5F10%5Fiitj%5Fac%5Fin%2FDocuments%2FAudio%5FAssignment%5FDL&ga=1)

## Network Architecture

### Architecture 1
- Use 1D-convolution for feature extraction.
- The base network should be at least three layers deep.
- Implement a fully-connected layer for multi-class classification.
- Free to use any number of layers, strides, kernel size, number of filters, activation functions, pooling, and other free parameters. Use PyTorch only.

### Architecture 2
- Use 1D-convolution for feature extraction. Same base network as above.
- Implement a transformer encoder network (from scratch) with a multi-head self-attention mechanism.
- Add the <cls> token.
- On top of the transformer, an MLP head should be there for classification.
- Model should have at least two attention blocks for number of heads = 1, 2, 4.
- Self-attention is to be implemented from scratch to solve a 10-class classification problem using PyTorch/PyTorch Lightning.

### Analysis
- Prepare a detailed analysis on which model achieves the best accuracy and why.

## Tasks

1. Train for 100 epochs. Plot accuracy and loss per epoch on Weight and Biases (WandB) platform. Mention plots in the report.
2. Perform k-fold validation, for k=4.
3. Prepare Accuracy, Confusion matrix, F1-scores, and AUC-ROC curve for the test set for all the combinations of the network. In-built functions can be used for this purpose.
4. Report total trainable and non-trainable parameters.
5. Perform hyper-parameter tuning and report the best hyper-parameter set.

Perform all tasks for both architectures with both test-split configurations and k-fold validation strategies.


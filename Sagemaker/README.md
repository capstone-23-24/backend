# RoBERTa Model Training and Deployment with Amazon SageMaker

## Overview

This project demonstrates how to train a RoBERTa-based model using PyTorch and Hugging Face Transformers and deploy it on Amazon SageMaker. The model is trained for a multi-label classification task using a custom dataset.

## Project Structure

- **roberta_model_class.py**: Defines the PyTorch model architecture (`MyModel` class).
- **roberta_dataset_class.py**: Implements the dataset structure (`MyDataset` class).
- **entry_point.py**: The main script for training and deploying the model on SageMaker.
- **Dockerfile**: Defines the Docker image for packaging the model and dependencies.
- **requirements.txt**: Lists the Python dependencies for the project.
- **<your_dataset>.csv**: Your custom dataset in CSV format.

## How to Use

### 1. Dataset Preparation

Prepare your dataset in CSV format with columns such as `casePart1`, `casePart2`, `label1`, ..., `label10`. Ensure that the dataset is appropriately labeled and split into training and validation sets.

### 2. Docker Image Build and Push

Build and push the Docker image to a container registry (e.g., Amazon ECR).

```bash
docker build -t your-image-name .
docker tag your-image-name:latest your-repository-uri:latest
docker push your-repository-uri:latest

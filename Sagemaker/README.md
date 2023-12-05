# RoBERTa Model Training and Deployment with Amazon SageMaker

## Overview

This project demonstrates how to train a RoBERTa-based model using PyTorch and Hugging Face Transformers and deploy it on Amazon SageMaker. The model is trained for a multi-label classification task using a custom dataset.

## Project Structure

- **roberta_model.py**: Defines the PyTorch model architecture (`MyModel` class).
- **roberta_dataset.py**: Implements the dataset structure (`MyDataset` class).
- **sagemaker_entry_point.py**: The main script for training and deploying the model on SageMaker.
- **dockerfile**: Defines the Docker image for packaging the model and dependencies.
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
```

## Docker file Info

### Working Directory in Dockerfile

The Dockerfile for this project establishes the working directory as `/opt/ml/model`. In the context of Amazon SageMaker, this directory is conventionally used to store model artifacts. During a SageMaker training job, the training data is provided in the `/opt/ml/input/data` directory, and SageMaker expects the trained model to be saved in the `/opt/ml/model` directory. Setting the working directory to `/opt/ml/model` ensures alignment with SageMaker conventions, facilitating seamless integration with the SageMaker environment. This setup allows for straightforward handling of input data and saving of model artifacts in a manner consistent with SageMaker expectations.


## SageMaker Estimator Configuration (sagemaker_estimator.py)

The `sagemaker_estimator.py` file is essential for configuring the Amazon SageMaker estimator to facilitate the training and deployment of our RoBERTa-based model. It defines the SageMaker PyTorch estimator using the SageMaker Python SDK. In this script, we set up key parameters such as the entry point script (`entry_point.py`), the instance type for training, the Docker image URI, and hyperparameters required for model training. The estimator serves as the blueprint for running the training job on SageMaker, orchestrating the model training process with the specified configurations. Once the training is complete, the SageMaker estimator is also used to deploy the trained model as a SageMaker endpoint for easy inference. This separation of configuration details into the `sagemaker_estimator.py` file enhances the clarity and modularity of the project structure.


## SageMaker Entry Point Script (sagemaker_entry_point.py)

The `sagemaker_entry_point.py` script serves as the main entry point for executing tasks during both the training and inference phases on Amazon SageMaker. This script is crucial for SageMaker Script Mode, enabling seamless integration with SageMaker's infrastructure. For model training, the script handles tasks such as loading the dataset, defining the model architecture, configuring training hyperparameters, and saving the trained model artifacts. During inference, the script is responsible for loading the deployed model, processing input data, and producing predictions. The careful design of this script ensures compatibility with SageMaker's requirements, allowing for efficient execution in SageMaker training jobs and inference endpoints. By encapsulating these functionalities in a dedicated entry point script, the project achieves a clean and modular structure, enhancing maintainability and ease of integration with the SageMaker environment.



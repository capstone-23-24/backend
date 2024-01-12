from sagemaker.pytorch import PyTorch
from sagemaker.huggingface import HuggingFace

# Specify your Docker image URI
docker_image_uri = '131750570751.dkr.ecr.us-east-1.amazonaws.com/capstone:latest'

# Create a PyTorch estimator
estimator = PyTorch(image_uri = docker_image_uri,
                    entry_point='sagemaker_entry_point.py',
                    role='arn:aws:iam::131750570751:role/service-role/AmazonSageMaker-ExecutionRole-20231120T210740',
                    framework_version='1.8.1',
                    py_version='py3',
                    instance_count=1,
                    instance_type='ml.m5.xlarge',  # Adjust instance type as needed
                    hyperparameters={'learning_rate': 0.001, 'batch_size': 32, 'epochs': 5},
                    output_path='s3://sagemaker-us-east-1-131750570751/Output/') # Our s3 bucket

estimator.fit({'train': 's3://sagemaker-us-east-1-131750570751/training_data.csv', 'test': 's3://sagemaker-us-east-1-131750570751/test_data.csv'})
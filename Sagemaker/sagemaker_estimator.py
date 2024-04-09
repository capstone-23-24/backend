from sagemaker.pytorch import PyTorch
from sagemaker.huggingface import HuggingFace

# Specify your Docker image URI
docker_image_uri = '381491935768.dkr.ecr.us-east-1.amazonaws.com/capstone'

# Create a PyTorch estimator
estimator = PyTorch(image_uri = docker_image_uri,
                    entry_point='sagemaker_entry_point.py',
                    role='arn:aws:iam::381491935768:role/AmazonSageMaker-ExecutionRole-20231120T210740',
                    framework_version='1.8.1',
                    py_version='py3',
                    instance_count=1,
                    instance_type='ml.m5.xlarge',  # Adjust instance type as needed
                    hyperparameters={'learning_rate': 0.001, 'batch_size': 8, 'epochs': 5},
                    output_path='s3://capstone-19283/output/') # Our s3 bucket

estimator.fit({'train': 's3://capstone-19283/training_data.csv', 'test': 's3://capstone-19283/test_data.csv'}, job_name="demo-search")
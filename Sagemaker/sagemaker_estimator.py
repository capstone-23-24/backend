from sagemaker.pytorch import PyTorch

# Specify your Docker image URI
docker_image_uri = 'your-docker-image-uri'

# Create a PyTorch estimator
estimator = PyTorch(entry_point='saggemaker_entry_point.py',
                    source_dir='path/to/your/script',
                    role=role,
                    framework_version='1.8.1',
                    py_version='py3',
                    instance_count=1,
                    instance_type='ml.p2.xlarge',  # Adjust instance type as needed
                    hyperparameters={'your_hyperparameter': 'value'},
                    output_path='s3://sagemaker-us-east-1-131750570751/Output/') # Our s3 bucket
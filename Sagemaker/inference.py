import os
import torch
import boto3
import tarfile
import gzip
from transformers import RobertaConfig, RobertaTokenizer
from sagemaker_inference import content_types, default_inference_handler, encoder
from roberta_model import MyModel  # Import the MyModel class from roberta_model_class.py
import urllib.parse

# Function to download and extract model from S3
def download_extract_model(s3_bucket, s3_object, local_tar_file, local_model_dir):
    os.makedirs(local_model_dir, exist_ok=True) 
    
    s3 = boto3.client('s3')
    s3.download_file(s3_bucket, s3_object, local_tar_file)

    with gzip.open(local_tar_file, 'rb') as f_in:
        with tarfile.open(fileobj=f_in, mode='r') as tar:
            tar.extractall(local_model_dir)
            local_model_dir = tar.getnames()
            print(f"Extracted Files: {local_model_dir}")

num_labels = 7
s3_bucket = 'sagemaker-us-east-1-131750570751'
s3_object = 'Output/capstone-2024-01-19-19-21-40-374/output/model.tar.gz'
local_tar_file = '/tmp/model.tar.gz'
local_model_dir = '/tmp/extracted_model_directory/'

download_extract_model(s3_bucket, s3_object, local_tar_file, local_model_dir)

# Load the configuration from config.json
s3_config_url= 's3://sagemaker-us-east-1-131750570751/extracted_model_directory//s3:/sagemaker-us-east-1-131750570751/Output/config.json'
local_config_file = '/tmp/config.json'
s3 = boto3.client('s3')
s3.download_file(s3_bucket, urllib.parse.urlparse(s3_config_url).path.lstrip('/'), local_config_file)
config = RobertaConfig.from_json_file(local_config_file)

# Initialize your model with the loaded configuration
model = MyModel(num_labels=num_labels)

s3_model_bin_key = 'extracted_model_directory//s3:/sagemaker-us-east-1-131750570751/Output/pytorch_model.bin'
local_model_bin_file = '/tmp/pytorch_model.bin'
s3.download_file(s3_bucket, s3_model_bin_key, local_model_bin_file)
model.load_state_dict(torch.load(local_model_bin_file, map_location=torch.device('cpu')))
print(model)

# Set the model to evaluation mode
model.eval()
class ModelHandler(default_inference_handler.DefaultInferenceHandler):
    def __init__(self, model):
        super(ModelHandler, self).__init__()
        self.model = model
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def default_input_fn(self, input_data, content_type):
        if content_type == content_types.JSON:
            input_text = input_data["text"]
        else:
            input_text = input_data.decode("utf-8")
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        return inputs

    def default_predict_fn(self, inputs, model):
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs

    def default_output_fn(self, prediction, accept):
        return str(prediction)

# Create an instance of the model handler
model_handler = ModelHandler(model)

# Define the input and output content types for the SageMaker endpoint
content_type = "application/json"
accept = "text/plain"

# Invoke the default_inference_handler's handler function
default_inference_handler.handler(model_handler, content_type, accept)
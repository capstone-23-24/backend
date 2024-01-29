import os
import torch
import boto3
import tarfile
import gzip
import logging
import shutil
from transformers import RobertaConfig, RobertaTokenizer
from sagemaker_inference import content_types, default_inference_handler
from roberta_model import MyModel  # Import the MyModel class from roberta_model_class.py
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to download and extract model from S3
def download_extract_model(s3_bucket, s3_object, local_tar_file, local_model_dir):
    try:
        # Create the target directory if it doesn't exist
        os.makedirs(local_model_dir, exist_ok=True)

        # Clear the target directory before downloading new files
        for filename in os.listdir(local_model_dir):
            file_path = os.path.join(local_model_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error('Failed to delete %s. Reason: %s', file_path, e)
                raise

        # Download the model tar.gz file from S3
        s3 = boto3.client('s3')
        logger.info(f"Downloading model from S3 bucket '{s3_bucket}'")
        s3.download_file(s3_bucket, s3_object, local_tar_file)

        # Extract the model files
        logger.info("Extracting model files")
        with gzip.open(local_tar_file, 'rb') as f_in:
            with tarfile.open(fileobj=f_in, mode='r') as tar:
                tar.extractall(local_model_dir)
                extracted_files = tar.getnames()  # Use a separate variable to store the names of extracted files
                logger.info(f"Extracted Files: {extracted_files}")

    except Exception as e:
        logger.error("Failed to download or extract model: %s", e)
        raise

class ModelHandler(default_inference_handler.DefaultInferenceHandler):
    logger.error("Initializing ModelHandler")
    
    def __init__(self, model):
        logger.error("Initializing __init__ function")
        super(ModelHandler, self).__init__()
        self.model = model
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def default_input_fn(self, input_data, content_type):
        logger.error("Preparing input data")
        if content_type == content_types.JSON:
            input_text = input_data["text"]
        else:
            input_text = input_data.decode("utf-8")
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        return inputs

    def default_predict_fn(self, inputs):
        logger.error("Making predictions")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs['logits']

    def default_output_fn(self, prediction, accept):
        logger.error("Preparing output data")
        return str(prediction)

num_labels = 7
s3_bucket = 'sagemaker-us-east-1-131750570751'
s3_object = 'Output/demo-search-3/output/model.tar.gz'
local_tar_file = '/tmp/model.tar.gz'
local_model_dir = '/tmp/extracted_model_directory/'

download_extract_model(s3_bucket, s3_object, local_tar_file, local_model_dir)

# Load the configuration from config.json
s3_config_url = 's3://sagemaker-us-east-1-131750570751/extracted_model_directory//s3:/sagemaker-us-east-1-131750570751/Output/config.json'
local_config_file = '/tmp/config.json'
try:
    s3 = boto3.client('s3')
    logger.error("Downloading model configuration")
    s3.download_file(s3_bucket, urllib.parse.urlparse(s3_config_url).path.lstrip('/'), local_config_file)
    config = RobertaConfig.from_json_file(local_config_file)
    logger.error("Model configuration loaded")
except Exception as e:
    logger.error("Failed to download or load model configuration: %s", e)
    raise

# Initialize your model with the loaded configuration
model = MyModel(num_labels=num_labels)

s3_model_bin_key = 'extracted_model_directory//s3:/sagemaker-us-east-1-131750570751/Output/pytorch_model.bin'
local_model_bin_file = '/tmp/pytorch_model.bin'
try:
    logger.error("Downloading model binary")
    s3.download_file(s3_bucket, s3_model_bin_key, local_model_bin_file)
    state_dict = torch.load(local_model_bin_file, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    adapted_dict = {('roberta.' + k): v for k, v in state_dict.items()}
    model.load_state_dict(adapted_dict)
    logger.error("Model loaded successfully")
except Exception as e:
    logger.error("Failed to download or load model binary: %s", e)
    raise

# Set the model to evaluation mode
model.eval()

# Create an instance of the model handler
model_handler = ModelHandler(model)

def handle(request, context):
    logger.error("Called handler function!")
    logger.debug("Request: " + request)
    logger.debug("Context: " + context)
    try:
        predictions = model_handler.default_predict_fn(request)
        return predictions
    except:
        logger.error("Unable to predict with given data!")
        return { "error": "unable to predict with given data!" }
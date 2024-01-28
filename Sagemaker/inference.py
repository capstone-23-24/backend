import os
import torch
import boto3
import tarfile
import gzip
import logging
from transformers import RobertaConfig, RobertaTokenizer
from sagemaker_inference import model_server, encoder, content_types
from roberta_model import MyModel  # Assuming this is your custom model class
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Function to download and extract model from S3
def download_extract_model(s3_bucket, s3_object, local_tar_file, local_model_dir):
    try:
        os.makedirs(local_model_dir, exist_ok=True)
        s3 = boto3.client('s3')
        logger.info(f"Downloading model from S3 bucket {s3_bucket}...")
        s3.download_file(s3_bucket, s3_object, local_tar_file)
        
        with gzip.open(local_tar_file, 'rb') as f_in:
            with tarfile.open(fileobj=f_in, mode='r') as tar:
                tar.extractall(path=local_model_dir)
                logger.info("Model extracted successfully to %s", local_model_dir)
    except Exception as e:
        logger.error("Failed to download or extract model: %s", e)
        raise

num_labels = 7
s3_bucket = 'sagemaker-us-east-1-131750570751'
s3_object = 'Output/capstone-2024-01-19-19-21-40-374/output/model.tar.gz'
local_tar_file = '/tmp/model.tar.gz'
local_model_dir = '/tmp/extracted_model_directory/'

download_extract_model(s3_bucket, s3_object, local_tar_file, local_model_dir)

# Load the configuration from config.json
s3_config_url = 's3://sagemaker-us-east-1-131750570751/Output/config.json'
local_config_file = '/tmp/config.json'
try:
    s3 = boto3.client('s3')
    logger.info("Downloading config file from %s", s3_config_url)
    s3.download_file(s3_bucket, urllib.parse.urlparse(s3_config_url).path.lstrip('/'), local_config_file)
    config = RobertaConfig.from_json_file(local_config_file)
    logger.info("Config file loaded successfully")
except Exception as e:
    logger.error("Error downloading or loading config file: %s", e)
    raise

# Initialize your model with the loaded configuration
model = MyModel(config=config)

s3_model_bin_key = 'Output/pytorch_model.bin'
local_model_bin_file = '/tmp/pytorch_model.bin'
try:
    logger.info("Downloading model binary from S3 key %s", s3_model_bin_key)
    s3.download_file(s3_bucket, s3_model_bin_key, local_model_bin_file)
    state_dict = torch.load(local_model_bin_file, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(state_dict)
    logger.info("Model binary loaded successfully")
except Exception as e:
    logger.error("Error downloading or loading model binary: %s", e)
    raise

# Set the model to evaluation mode
model.eval()
logger.info("Model set to evaluation mode")

class ModelHandler(default_inference_handler.DefaultInferenceHandler):
    def __init__(self, model):
        super(ModelHandler, self).__init__()
        self.model = model
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def default_input_fn(self, input_data, content_type):
        try:
            logger.info("Processing input data")
            if content_type == content_types.JSON:
                input_text = input_data["text"]
            else:
                input_text = input_data.decode("utf-8")
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            return inputs
        except Exception as e:
            logger.error("Error in input function: %s", e)
            raise

    def default_predict_fn(self, inputs, model):
        try:
            logger.info("Making predictions")
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs
        except Exception as e:
            logger.error("Error in predict function: %s", e)
            raise

    def default_output_fn(self, prediction, accept):
        try:
            logger.info("Formatting prediction output")
            return encoder.encode(prediction, accept)
        except Exception as e:
            logger.error("Error in output function: %s", e)
            raise

# Create an instance of the model handler
model_handler = ModelHandler(model)

# Start the model server with our handler
logger.info("Starting model server")
model_server.start_model_server(handler_service=model_handler

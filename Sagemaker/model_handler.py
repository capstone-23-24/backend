import os
import torch
import boto3
import tarfile
import tempfile
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
logger.setLevel(logging.INFO)

# Initialize S3 client
s3_client = boto3.client('s3')

def clear_s3_bucket(target_s3_bucket, target_s3_prefix):

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(target_s3_bucket)

    # Iterate through objects in the specified folder
    for obj in bucket.objects.filter(Prefix=target_s3_prefix):
        key = obj.key
        if not key.endswith('.txt'):
            obj.delete()

    # List all objects in the specified prefix
    # objects_to_delete = s3_client.list_objects_v2(Bucket=target_s3_bucket, Prefix=target_s3_prefix)
    # logger.error(f"Objects to be deleted: {objects_to_delete}")

    # # Delete the objects found in the prefix
    # if 'Contents' in objects_to_delete:
    #     delete_keys = {'Objects': []}
    #     delete_keys['Objects'] = [{'Key': obj['Key']} for obj in objects_to_delete['Contents']]
    #     s3_client.delete_objects(Bucket=target_s3_bucket, Delete=delete_keys)
    #     logger.error(f"Cleared {target_s3_prefix} in {target_s3_bucket}")

def download_extract_upload(s3_bucket, s3_object, target_s3_prefix, local_tar_file):
    # First, clear the target S3 prefix
    # clear_s3_bucket(s3_bucket, target_s3_prefix)

    # Download the tar.gz file from S3 to /tmp
    s3_client.download_file(s3_bucket, s3_object, local_tar_file)
    logger.error(f"Downloaded {s3_object} from {s3_bucket} to {local_tar_file}")

    with gzip.open(local_tar_file, 'rb') as f_in:
        with tarfile.open(fileobj=f_in, mode='r') as tar:
            tar.extractall('/tmp/' + target_s3_prefix)
            local_model_dir = tar.getnames()
            logger.error(f"Uploaded {local_model_dir} to {s3_bucket}")


    # # Extract the tar.gz file
    # with tarfile.open(local_tar_path, "r:gz") as tar:
    #     tar.extractall(path=tempfile.gettempdir())

    #     # Upload extracted files to the target S3 location
    #     for member in tar.getmembers():
    #         file_path = os.path.join(tempfile.gettempdir(), member.name)
    #         if os.path.isfile(file_path):
    #             s3_key = os.path.join(target_s3_prefix, member.name)
    #             s3_client.upload_file(file_path, s3_bucket, s3_key)
    #             logger.info(f"Uploaded {s3_key} to {s3_bucket}")
            
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
s3_extracted_folder_prefix = 'extracted_model_directory/'

download_extract_upload(s3_bucket, s3_object, s3_extracted_folder_prefix, local_tar_file)

# Load the configuration from config.json
s3_config_url = 's3://sagemaker-us-east-1-131750570751/extracted_model_directory/Config.json'
local_config_file = '/tmp/config.json'
try:
    logger.error("Downloading model configuration")
    s3_client.download_file(s3_bucket, urllib.parse.urlparse(s3_config_url).path.lstrip('/'), local_config_file)
    config = RobertaConfig.from_json_file(local_config_file)
    logger.error("Model configuration loaded")
except Exception as e:
    logger.error("Failed to download or load model configuration: %s", e)
    raise

# Initialize your model with the loaded configuration
model = MyModel(num_labels=num_labels)

s3_model_bin_key = 'extracted_model_directory/pytorch_model.bin'
local_model_bin_file = '/tmp/pytorch_model.bin'
try:
    logger.error("Downloading model binary")
    s3_client.download_file(s3_bucket, s3_model_bin_key, local_model_bin_file)
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

import os
import torch
import boto3
import tarfile
import gzip
import logging
import chardet
import traceback
import json
from transformers import RobertaConfig, RobertaTokenizer, RobertaTokenizerFast, RobertaForTokenClassification
from sagemaker_inference import content_types, default_inference_handler
from roberta_model import MyModel  # Import the MyModel class from roberta_model_class.py
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
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

def parse_bytearray(input):
    encoding = chardet.detect(input)['encoding']
    return json.loads(input.decode(encoding).replace("'", "\""))

def download_extract_model(s3_bucket, s3_object, local_tar_file, local_model_dir):
    os.makedirs(local_model_dir, exist_ok=True) 

    s3 = boto3.client('s3')
    s3.download_file(s3_bucket, s3_object, local_tar_file)

    with gzip.open(local_tar_file, 'rb') as f_in:
        with tarfile.open(fileobj=f_in, mode='r') as tar:
            tar.extractall(local_model_dir)
            local_model_dir = tar.getnames()
            logger.info(f"Extracted Files: {local_model_dir}")
            

class ModelHandler(default_inference_handler.DefaultInferenceHandler):
    # Other parts of the class remain unchanged...

    def __init__(self, model):
        super(ModelHandler, self).__init__()
        self.model = model
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    def default_input_fn(self, input_data, content_type):
        logger.info("Preparing input data")
        input_text = input_data["text"] if content_type == content_types.JSON else input_data.decode("utf-8")
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        return inputs

    def default_predict_fn(self, inputs):
        logger.info("Making predictions")
        self.model.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            outputs = self.model(**inputs)
        return {'logits': outputs['logits'], 'input_ids': inputs['input_ids']}


    def default_output_fn(self, prediction, accept=content_types.JSON):
        logger.info("Preparing output data")

        # Assuming 'prediction' is a dictionary that includes both logits and input IDs
        logits = prediction["logits"].detach().cpu().numpy()
        input_ids = prediction["input_ids"].cpu().numpy() 

        results = []
        for idx, logit in enumerate(logits):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[idx])
            label_indices = logit.argmax(axis=1)
            print(f"label_indices: {label_indices}")
            labels = [self.model.config.id2label[label_id] for label_id in label_indices]

            entities = []
            current_entity = None
            logger.info(f"Tokens: {tokens}")
            logger.info(f"Labels: {labels}")
            for token, label in zip(tokens, labels):
                if label in ['PERSON', 'LOCATION', 'GENDER']:
                    if current_entity and current_entity["entity"] == label:
                        current_entity["text"] += " " + token
                    else:
                        if current_entity:
                            entities.append(current_entity)
                        current_entity = {"entity": label, "text": token}
                else:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None

            if current_entity:
                entities.append(current_entity)

            results.append({"entities": entities})

        if accept.lower() == content_types.JSON:
            response = results
            return [response]
        else:
            raise Exception(f'Requested unsupported ContentType in Accept: {accept}')
       
num_labels = 7
s3_bucket = 'capstone-19283'
s3_object = 'output/demo-search-ner-6/output/model.tar.gz'
local_tar_file = '/tmp/model.tar.gz'
local_model_dir = '/tmp/extracted_model_directory/'

download_extract_model(s3_bucket, s3_object, local_tar_file, local_model_dir)

# Load the configuration from config.json
s3_config_url = 's3://capstone-19283/extracted_model_directory/s3:/sagemaker-us-east-1-131750570751/Output/config.json'
local_config_file = '/tmp/config.json'
try:
    logger.info("Downloading model configuration")
    s3_client.download_file(s3_bucket, urllib.parse.urlparse(s3_config_url).path.lstrip('/'), local_config_file)
    config = RobertaConfig.from_json_file(local_config_file)
    logger.info("Model configuration loaded")
except Exception as e:
    logger.error("Failed to download or load model configuration: %s", e)
    raise

# Initialize your model with the loaded configuration
model = MyModel(num_labels=num_labels)

s3_model_bin_key = 'extracted_model_directory/s3:/sagemaker-us-east-1-131750570751/Output/pytorch_model.bin'
local_model_bin_file = '/tmp/pytorch_model.bin'
try:
    logger.info("Downloading model binary")
    s3_client.download_file(s3_bucket, s3_model_bin_key, local_model_bin_file)
    state_dict = torch.load(local_model_bin_file, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    adapted_dict = {('roberta.' + k): v for k, v in state_dict.items()}
    model.load_state_dict(adapted_dict)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error("Failed to download or load model binary: %s", e)
    raise

# Set the model to evaluation mode
model.eval()

# Create an instance of the model handler
model_handler = ModelHandler(model)

def handle(request, context):
    logger.info("Called handler function!")
    logger.info(f"Request: {str(request)}")
    logger.info(f"Context: {str(context)}")
    try:
        input = model_handler.default_input_fn(parse_bytearray(request[0]["body"]), content_type=content_types.JSON)
        predictions = model_handler.default_predict_fn(input)
        output = model_handler.default_output_fn(predictions, content_types.JSON)
        return output
    except Exception as err:
        error_str = traceback.format_exc()
        logger.error(f"Unable to predict with given data due to: {error_str}")
        logger.debug(error_str)
        return [{ "status": 500, "message": "unable to predict with given data!", "error": str(err) }]

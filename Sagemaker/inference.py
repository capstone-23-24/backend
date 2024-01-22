import torch
import boto3
import tarfile
from transformers import RobertaConfig, RobertaTokenizer
from sagemaker_inference import content_types, default_inference_handler, encoder
from roberta_model import MyModel  # Import the MyModel class from roberta_model_class.py

# Function to download and extract model from S3
def download_extract_model(s3_bucket, s3_object, local_tar_file):
    s3 = boto3.client('s3')
    s3.download_file(s3_bucket, s3_object, local_tar_file)

    with tarfile.open(local_tar_file) as tar:
        tar.extractall()

# Download and extract the model (adjust the paths as necessary)
s3_bucket = 'sagemaker-us-east-1-131750570751'
s3_object = 'Output/capstone-2024-01-19-19-21-40-374/output/model.tar.gz'
local_tar_file = 'model.tar.gz'
download_extract_model(s3_bucket, s3_object, local_tar_file)

# Path to the extracted model files
local_model_dir = 'extracted_model_directory'  # Update this with the directory name inside the tarball

# Load the configuration from config.json
config = RobertaConfig.from_json_file(f"{local_model_dir}/config.json")

# Initialize your model with the loaded configuration
model = MyModel(config)

# Load the weights from pytorch_model.bin
model.load_state_dict(torch.load(f"{local_model_dir}/pytorch_model.bin", map_location=torch.device('cpu')))

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

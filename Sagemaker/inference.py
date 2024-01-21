import torch
from transformers import RobertaTokenizer
from sagemaker_inference import content_types, default_inference_handler, encoder
from roberta_model import MyModel  # Import the MyModel class from roberta_model_class.py

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

# Load your trained model
model = MyModel.from_pretrained("s3://sagemaker-us-east-1-131750570751/Output/capstone-2024-01-19-19-21-40-374/output/model.tar.gz")
model.eval()

# Create an instance of the model handler
model_handler = ModelHandler(model)

# Define the input and output content types for the SageMaker endpoint
content_type = "application/json"
accept = "text/plain"

# Invoke the default_inference_handler's handler function
default_inference_handler.handler(model_handler, content_type, accept)

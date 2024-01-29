import model_handler
import subprocess
from sagemaker_inference import model_server

def main():
    print("--- Starting Model Server ---")
    # Start the model server with our handler
    model_server.start_model_server(handler_service=model_handler.__file__ + ":handle")
    subprocess.call(["tail", "-f", "/dev/null"])

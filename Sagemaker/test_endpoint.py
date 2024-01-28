import boto3

# Create a SageMaker runtime client
client = boto3.client('runtime.sagemaker')

# Specify your endpoint name
endpoint_name = 'demo-search17'

# The data payload you want to send for inference
data = '{"data": "I am 22 years old and I am a student at the University of California, Berkeley."}'

# Call the endpoint
response = client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',  # Adjust based on the expected content type by your model
    Body=data
)

# Process the response
print(response['Body'].read())

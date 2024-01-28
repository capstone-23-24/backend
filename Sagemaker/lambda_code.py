import os
import boto3
import json
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # You can adjust this to DEBUG for more verbose output

# Grab environment variables
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
logger.info(f"Using SageMaker Endpoint: {ENDPOINT_NAME}")

runtime = boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    logger.info("Received event: %s", json.dumps(event, indent=2))

    try:
        data = json.loads(json.dumps(event))
        payload = data['data']

        # Log the payload being sent to the SageMaker endpoint
        logger.info("Payload sent to SageMaker endpoint: %s", payload)

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Body=payload,
        )

        results = json.loads(response['Body'].read().decode())
        
        # Log the results received from the SageMaker endpoint
        logger.info("Results received from SageMaker endpoint: %s", results)
        
        return results

    except Exception as e:
        logger.error("Error processing the Lambda function", exc_info=True)
        raise e  # Rethrowing the exception after logging it

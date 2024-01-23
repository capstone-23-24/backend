import json
import os
import torch
import numpy as np
from roberta_model import MyModel

def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    """
    model = MyModel(num_labels=7)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'pytorch_model.bin')))
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == "text/json":
        return request_body["Text"]
    else:
        raise ValueError(
            "Content type {} is not supported.".format(request_content_type)
        )


def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    feature_contribs = model.predict(input_data, pred_contribs=True, validate_features=False)
    print(prediction)
    print(feature_contribs)
    output = np.hstack((prediction[:, np.newaxis], feature_contribs))
    return output


def output_fn(predictions, content_type):
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    """

    return predictions

    # if content_type == "text/csv":
    #     return ','.join(str(x) for x in predictions[0])
    # else:
    #     raise ValueError("Content type {} is not supported.".format(content_type))
# major part of code sourced from aws sagemaker example:
# https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/scikit_bring_your_own/container/decision_trees/predictor.py

import io
import os
import sys
import traceback
import warnings
from typing import List

import pandas as pd
from fastapi import Body, FastAPI

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")

import algorithm.utils as utils
from algorithm.model.classifier import MODEL_NAME
from algorithm.model_server import ModelServer

prefix = "/opt/ml_vol/"
data_schema_path = os.path.join(prefix, "inputs", "data_config")
model_path = os.path.join(prefix, "model", "artifacts")
failure_path = os.path.join(prefix, "outputs", "errors", "serve_failure")


# get data schema - its needed to set the prediction field name
# and to filter df to only return the id and pred columns
data_schema = utils.get_data_schema(data_schema_path)


# initialize your model here before the app can handle requests
model_server = ModelServer(model_path=model_path, data_schema=data_schema)


# The fastapi app for serving predictions
app = FastAPI()


@app.get("/ping", tags=["ping", "healthcheck"])
async def ping() -> dict:
    """Determine if the container is working and healthy."""
    response = f"Hello, I am {MODEL_NAME} model and I am at you service!"
    return {
        "success": True,
        "message": response,
    }


@app.post("/infer", tags=["inference"])
async def infer(instances: List[dict] = Body(embed=True)) -> dict:
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    # Do the prediction
    try:
        data = pd.DataFrame.from_records(instances)
        print(f"Invoked with {data.shape[0]} records")
        print(data)
        predictions = model_server.predict_to_json(data)
        return {
            "predictions": predictions,
        }
    except Exception as err:
        # Write out an error file. This will be returned as the failureReason to the client.
        trc = traceback.format_exc()
        with open(failure_path, "w") as s:
            s.write("Exception during inference: " + str(err) + "\n" + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print("Exception during inference: " + str(err) + "\n" + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        return {
            "success": False,
            "message": f"Exception during inference: {str(err)} (check failure file for more details)",
        }

import os
import pandas as pd
import time
import multiprocessing
from tqdm.contrib.concurrent import thread_map
from contextlib import closing
import logging
from gcp_functions import GCPFunctions

REQUIRED_ARGS = [
  "PROJECT_ID",
  "INPUT_BUCKET",
  "INPUT_FOLDER",
  "OUTPUT_BUCKET",
  "OUTPUT_FOLDER"
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_request_to_dict(request: flask.Request):
    request_json = request.get_json(silent=True)
    for arg in REQUIRED_ARGS:
        if arg not in request.args:
            return UserWarning("Did not pass the required arg: {arg}")
    return request_json.args

@functions_framework.http
def run_preproc(request: flask.Request):
    """
    API request example:
      {
          "PROJECT_ID": "gcp_project",
          "INPUT_BUCKET": "input_bucket",
          "INPUT_FOLDER" : "input_folder",
          "OUTPUT_BUCKET" : "output_bucket",
          "OUTPUT_FOLDER" : "output_folder",
      }
    """
    start_time = time.time()
    request_dict = get_request_to_dict(request=request)
    gcp_functions = GCPFunctions(request=request_dict)
    gcp_functions.create_input_metadata()
    
    

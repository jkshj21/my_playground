from typing import Dict
import logging
from tqdm import tqdm as tq
from google.cloud import storage

logging.basicConfig(
    format="[dfcx] %(levelname)s:%(message)s", level=logging.INFO
)

class GCPFunctions():
    """
    """
    def __init__(
        self,
        request: Dict,
        url_prefix: str,
        id_prefix: str
    ):
        self.project_id = request.get("PROJECT_ID")
        self.input_bucket = request.get("INPUT_BUCKET")
        self.input_folder = request.get("INPUT_FOLDER")
        self.output_bucket = request.get("outPUT_BUCKET")
        self.output_folder = request.get("OUTPUT_FOLDER")
        self.url_prefix = url_prefix
        self.id_prefix = id_prefix

    def create_json_str(
        blob,
        blob_number,
        blob_str,
    ):
        _id = f"{self.id_prefix}_{blob_number}"
        path, ext = os.path.splitext(blob.name)
        name = os.path.basename(path)
        url_postfix = os.path.join(*name.split(".")[:-1]) if str(name).endswith("mainContent") else os.path.join(*name.split("."))
        url = f"{self.url_prefix}/{url_postfix}{ext}"
        uri = f"gs://{bucket_name}/{blob_obj.name}"
        mime_type = "text/plain"
        if str(blob.name).endswith(".html"):
            mime_type = "text/html"
        elif str(blob.name).endswith(".pdf"):
            mime_type = "application/pdf"
        json_dict = {
            "id": _id,
            "jsonData": {"title": title, "url": url},
            "content": {"mimeType": mime_type, "uri": uri}
        }
        response = json.dumps(json_dict)
        logging.info(response)
        return response

    def import_metadata(
        json_str,
        destination_bucket,
        destination_folder
    ):
        path = os.path.join(destination_folder, "metadata.jsonl")
        blob_obj = destination_bucket.blob(path)
        blob_obj.upload_from_string(json_str)
        
    def create_input_metadata()
        storage_client = storage.Client(project = self.project_id)
        source_bucket = storage_client.get_bucket(self.input_folder)
        blobs = [b for b in source_bucket.list_blobs(prefix=self.input_folder, delimiter="")][:limit]
        json_list = []
        for i, obj in enumerate(tq(blobs, desc = "Writing Metadata in input bucket..")):
        # Skip files that are not html, txt, json, or pdf
            if not(
                str(obj.name).endswith(".html")
                or str(obj.name).endswith(".txt")
                or str(obj.name).endswith(".pdf")
            ):
                continue
            # Download the HTML for extracting the header if necessary
            obj_str = obj.download_as_string()
            json_list.append(self.create_json_str(
                blob=obj,
                bucket_name=self.input_bucket,
                obj_number=i,
                blob_str=obj_str,
            ))
        json_str = "\n".join(json_list)
        self.import_metadata(
            json_str=json_str,
            destination_bucket=self.input_bucket,
            destination_folder=self.input_folder
        )
        
        
        

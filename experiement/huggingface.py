from huggingface_hub import HfApi, HfFolder, Repository

# Upload using huggingface_hub
from huggingface_hub import upload_folder

upload_folder(
    folder_path="./artifact/model",
    repo_id="JinOhara/intent-based-classifier",
    repo_type="model"
)

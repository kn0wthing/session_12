from huggingface_hub import HfApi
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def upload_to_hub(
    model_path,
    repo_name,
    organization=None,  # Optional: your organization name
    token=None  # Token will be loaded from environment variable
):
    # Get token from environment variable
    token = token or os.getenv('HF_TOKEN')
    if not token:
        raise ValueError("No Hugging Face token found. Please set HF_TOKEN environment variable.")

    # Initialize the HF API
    api = HfApi()
    
    # Create the repo URL
    repo_id = f"{organization}/{repo_name}" if organization else repo_name
    
    # Create the repository
    api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
        repo_type="model",
        private=False,
        token=token
    )
    
    # Upload all files
    files_to_upload = [
        ("README.md", "README.md"),
        (model_path, "gpt-config.pth"),
        ("requirements.txt", "requirements.txt"),
        ("model.py", "model.py"),
    ]
    
    for local_file, repo_path in files_to_upload:
        try:
            api.upload_file(
                path_or_fileobj=local_file,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="model",
                token=token
            )
        except Exception as e:
            print(f"Error uploading {local_file}: {e}")

if __name__ == "__main__":
    upload_to_hub(
        model_path="gpt-config.pth",
        repo_name="custom-gpt-model"
    ) 
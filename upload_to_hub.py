from huggingface_hub import HfApi
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch

def upload_to_hub(
    model_path,
    repo_name,
    organization=None,  # Optional: your organization name
    token="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"  # Your HF token
):
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
    
    # Initialize the repository with a README if it doesn't exist
    try:
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
    except Exception as e:
        print(f"Initial README upload: {e}")
    
    # Upload the model files
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="gpt-config.pth",
        repo_id=repo_id,
        repo_type="model",
        token=token
    )
    
    # Upload all other files
    files_to_upload = [
        "requirements.txt",
        "model.py",
    ]
    
    for file in files_to_upload:
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=file,
            repo_id=repo_id,
            repo_type="model",
            token=token
        )

if __name__ == "__main__":
    upload_to_hub(
        model_path="gpt-config.pth",
        repo_name="custom-gpt-model",  # Changed to a more specific name
        token="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"
    ) 
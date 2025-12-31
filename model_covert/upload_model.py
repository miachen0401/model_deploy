"""
Upload a local Hugging Face model folder to the Hugging Face Hub.
"""

from huggingface_hub import HfApi


def upload_model_folder(
    folder_path: str,
    repo_id: str,
    commit_message: str = "Upload fine-tuned model",
) -> None:
    """
    Upload an entire folder (config/tokenizer/weights) to a HF Hub model repo.

    Args:
        folder_path: Local path to the exported HF folder.
        repo_id: Repo id in the form "username/repo_name".
        commit_message: Commit message for the upload.
    """
    api = HfApi()
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )


if __name__ == "__main__":
    upload_model_folder(
        folder_path="../RL_model/Qwen1.5B/news",
        repo_id="Hula0401/qwen-1.5b-finetuned-news1.0",
        commit_message="Upload converted Qwen 1.5B fine-tuned model in HF format",
    )

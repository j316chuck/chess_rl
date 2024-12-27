import os 

from huggingface_hub import login, HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM
from searchless_chess.src.chess_tokens import get_all_uci_chess_moves

def upload_s3_path_awscli(
    upload_path: str,
    s3_uri: str,
    max_concurrent_requests: int = 200,
    quiet: bool = False,
    **kwargs,
):  
    import re
    import subprocess

    if not re.match("s3:.*", s3_uri):
        raise ValueError

    s3_uri = s3_uri.rstrip("/")
    flags = []
    if "AWS_PROFILE" in os.environ:
        flags += ["--profile", os.environ["AWS_PROFILE"]]

    kwargs = {}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL

    subprocess.run(
        [
            "aws",
            "configure",
            "set",
            "s3.max_concurrent_requests",
            str(max_concurrent_requests),
            *flags,
        ],
        check=True,
    )
    recursive = "--recursive" if os.path.isdir(upload_path) else ""
    subprocess.run(["aws", "s3", "cp", recursive, upload_path, s3_uri, *flags], **kwargs, check=True)


if __name__ == "__main__":
    S3_UPLOAD_PATH = os.environ["S3_UPLOAD_PATH"]
    TOKEN=os.environ["HF_TOKEN"]
    models = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
    ]
    local_save_dir = "./models"
    os.makedirs(local_save_dir, exist_ok=True)

    # HuggingFace authentication
    login(token=TOKEN)  # Replace with your HuggingFace token

    for base_model_name in models:
        # Define the model ID and tokenizer
        updated_model_name = f"j316chuck/{base_model_name}-Chess"

        # Load the base tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        print(f"Loaded {base_model_name} model and tokenizer")
        # Get the UCI chess moves
        chess_moves = get_all_uci_chess_moves()
        assert len(chess_moves) == 4164, "Expected 4164 chess moves, got {}".format(len(chess_moves))

        # Extend the tokenizer with new tokens (chess moves)
        new_tokens = [move for move in chess_moves if move not in tokenizer.vocab]
        tokenizer.add_tokens(new_tokens)
        print("New tokenizer size:", len(tokenizer))
        # Resize the model's embedding layer to match the new tokenizer size
        old_size = len(model.get_input_embeddings())
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized model's embedding layer from {old_size} to {len(model.get_input_embeddings())}")
        # Save the updated model and tokenizer locally
        model.save_pretrained(local_save_dir)
        tokenizer.save_pretrained(local_save_dir)
        print(f"Saved model and tokenizer to {local_save_dir}")
        print(os.listdir(local_save_dir))
        
        # Upload to the HuggingFace Hub
        api = HfApi()
        api.create_repo(repo_id=updated_model_name, repo_type="model", private=True, exist_ok=True)
        tokenizer.push_to_hub(updated_model_name)
        print("Pushed tokenizer to HuggingFace Hub")
        model.push_to_hub(updated_model_name)
        print("Pushed model to HuggingFace Hub")
        print(f"Model successfully uploaded to: https://huggingface.co/{updated_model_name}")


    upload_s3_path_awscli(local_save_dir, S3_UPLOAD_PATH)

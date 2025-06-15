import argparse
from huggingface_hub import snapshot_download

def download(model_name, output_dir):
    print(f"Downloading {model_name} to {output_dir}")
    snapshot_download(
        repo_id=model_name,
        local_dir=output_dir
    )
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    download(args.model, args.output_dir)
    

from huggingface_hub import snapshot_download
import os

def main():
    # Create directory for model weights
    os.makedirs("model_weights", exist_ok=True)
    
    # Download the model weights
    print("Downloading DeepSeek 7B model weights...")
    print("This will take some time depending on your internet speed (approximately 14GB download)")
    
    try:
        snapshot_download(
            repo_id="deepseek-ai/deepseek-llm-7b-base",
            local_dir="model_weights",
            local_dir_use_symlinks=False
        )
        print("Download complete! The model weights are in the 'model_weights' directory.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Make sure you have enough disk space (need at least 20GB free)")
        print("3. Try using a VPN if the download is too slow")
        print("4. You can also download manually from https://huggingface.co/deepseek-ai/deepseek-llm-7b-base")

if __name__ == "__main__":
    main()

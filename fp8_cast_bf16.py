import argparse
import os
from tqdm import tqdm
import torch
import shutil
from kernel import weight_dequant

def convert_model(input_path: str, output_path: str):
    """Convert FP8 model weights to BF16 format."""
    print(f"\nConverting model from {input_path} to {output_path}")
    print("This may take several minutes...")
    
    try:
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Copy non-weight files
        print("\nCopying model configuration files...")
        for item in os.listdir(input_path):
            src = os.path.join(input_path, item)
            dst = os.path.join(output_path, item)
            if os.path.isfile(src) and not item.endswith('.safetensors'):
                shutil.copy2(src, dst)
        
        # Convert weights
        print("\nConverting model weights...")
        for filename in tqdm(os.listdir(input_path)):
            if filename.endswith('.safetensors'):
                input_file = os.path.join(input_path, filename)
                output_file = os.path.join(output_path, filename)
                
                # Load weights
                state_dict = torch.load(input_file)
                
                # Convert weights
                new_state_dict = {}
                for key, tensor in state_dict.items():
                    if tensor.dtype == torch.float8_e4m3fn:
                        # Get scale file
                        scale_key = f"{key}_scale"
                        if scale_key in state_dict:
                            scale = state_dict[scale_key]
                            # Dequantize weights
                            tensor = weight_dequant(tensor, scale)
                        else:
                            print(f"Warning: Scale not found for {key}")
                    
                    # Convert to BF16
                    tensor = tensor.to(torch.bfloat16)
                    new_state_dict[key] = tensor
                
                # Save converted weights
                torch.save(new_state_dict, output_file)
        
        print("\nConversion completed successfully!")
        print(f"Converted model saved to: {output_path}")
        
    except Exception as e:
        print(f"\nError during conversion: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure you have enough disk space")
        print("2. Verify input model path is correct")
        print("3. Check if CUDA is available for tensor operations")
        print("4. Make sure you have write permissions in output directory")
        raise e

def main():
    parser = argparse.ArgumentParser(description="Convert FP8 model weights to BF16")
    parser.add_argument("--input-fp8-hf-path", type=str, required=True,
                      help="Path to input FP8 model weights")
    parser.add_argument("--output-bf16-hf-path", type=str, required=True,
                      help="Path to save converted BF16 weights")
    args = parser.parse_args()
    
    convert_model(args.input_fp8_hf_path, args.output_bf16_hf_path)

if __name__ == "__main__":
    main()

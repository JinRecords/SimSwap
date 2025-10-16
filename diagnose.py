from codeformer import CodeFormer
import torch

print("--- CodeFormer Diagnostic Tool ---")

try:
    # Determine device, fallback to CPU if CUDA is not available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    codeformer_net = CodeFormer()
    codeformer_net = codeformer_net.to(device)  # move model to CUDA or CPU
    print(f"Attempting to initialize CodeFormer on device: {device}")

    # Initialize the object using the method that worked

    print("\n[SUCCESS] CodeFormer object initialized.")
    print("-------------------------------------------------")
    print("Below is a list of all available attributes and methods:")
    print("Look for a name like 'process', 'enhance', 'run', 'forward', etc.\n")

    # Print all available attributes
    print(dir(codeformer_net))

    print("-------------------------------------------------")

except Exception as e:
    print(f"\n[ERROR] An error occurred: {e}")

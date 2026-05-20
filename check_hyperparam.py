import torch

# Path to your exported weights
weights_path = r"runs\platform-seg-rf.pt"

try:
    # Load the checkpoint dictionary
    # map_location='cpu' ensures it works even if you don't have a GPU right now
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)

    # 1. Check if it's a YOLOv8 / Ultralytics model
    if 'train_args' in ckpt:
        print("--- Training Hyperparameters Found! ---")
        args = ckpt['train_args']
        
        # Print common specific params you might care about
        # print(f"Epochs:      {args.get('epochs')}")
        # print(f"Batch Size:  {args.get('batch')}")
        # print(f"Img Size:    {args.get('imgsz')}")
        # print(f"Optimizer:   {args.get('optimizer')}")
        # print(f"Learning Rt: {args.get('lr0')}")
        
        # Uncomment to see EVERYTHING:
        print(args)
        
    # 2. Check if it's a standard PyTorch/RF-DETR checkpoint
    elif 'args' in ckpt:
        print("--- Model Arguments Found (Likely RF-DETR/Other) ---")
        print(ckpt['args'])
        
    else:
        print("Could not find standard 'train_args' or 'args' key.")
        print("Available keys in .pt file:", ckpt.keys())

except Exception as e:
    print(f"Error loading file: {e}")
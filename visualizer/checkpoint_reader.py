import torch

CHECKPOINT_FILE_PATH = "visualizer/ckpt_Nornal_MobileNetV2_original_79.98.pth"

# Load checkpoint to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(CHECKPOINT_FILE_PATH, map_location=device)

print(f"Best accuracy: {checkpoint['acc']}")
print(f"Best epoch: {checkpoint['epoch']}")

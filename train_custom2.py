import os
from pathlib import Path
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from anomalib.models import Patchcore
from anomalib.engine import Engine

# -----------------------------
# Custom Lightning DataModule
# -----------------------------
class CustomDataModule(LightningDataModule):
    def __init__(self, train_dataset, test_dataset=None, batch_size=2, category="zfill"):
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.category = category  # Required by Anomalib Engine

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset, batch_size=self.batch_size)
        return None

# -----------------------------
# Paths
# -----------------------------
train_dir = "./datasets/Z-Axis/zfill/train"
test_dir = "./datasets/Z-Axis/zfill/test"

# -----------------------------
# Transforms
# -----------------------------
transform = transforms.Compose([transforms.ToTensor()])

# -----------------------------
# Datasets
# -----------------------------
# ImageFolder expects a folder with subfolders for classes
train_dataset = ImageFolder(root=train_dir, transform=transform)
test_dataset = ImageFolder(root=test_dir, transform=transform)

print(f"Number of training images: {len(train_dataset)}")
print("Train classes:", train_dataset.classes)
print(f"Number of test images: {len(test_dataset)}")
print("Test classes:", test_dataset.classes)

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# Model & Engine
# -----------------------------
model = Patchcore()

# Create a workspace folder for logs, checkpoints
workspace_dir = Path("workspace")
workspace_dir.mkdir(exist_ok=True)

engine = Engine(
    task="classification",
    device=device,
    work_dir=workspace_dir  # Required for Engine to save logs/checkpoints
)

# -----------------------------
# Train the model
# -----------------------------
datamodule = CustomDataModule(train_dataset=train_dataset, test_dataset=test_dataset, category="zfill")
engine.train(datamodule=datamodule, model=model)

# -----------------------------
# Save checkpoint manually
# -----------------------------
ckpt_path = workspace_dir / "zfill_patchcore.ckpt"
torch.save(model.state_dict(), ckpt_path)
print(f"Checkpoint saved: {ckpt_path}")

# -----------------------------
# Optional: Evaluate on test images
# -----------------------------
# engine.evaluate(model=model, datamodule=datamodule)

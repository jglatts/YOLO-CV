import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader


if __name__ == "__main__":
    train_dir = "./datasets/Z-Axis/zfill/train/good"
    test_dir = "./datasets/Z-Axis/zfill/test/bad"

    # Transform images to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # PyTorch dataset for training
    train_dataset = ImageFolder(
        root=os.path.dirname(train_dir),  # ImageFolder expects a folder with subfolders
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    print(f"Number of training images: {len(train_dataset)}")
    print("Train classes:", train_dataset.classes)

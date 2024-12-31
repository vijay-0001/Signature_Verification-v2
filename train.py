import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import os
from PIL import Image
import config
from tqdm import tqdm  # Import tqdm for progress bars

# Define device (CUDA if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data preprocessing and loading
class SiameseDataset:
    def __init__(self, csv_file, image_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file, header=None)
        self.data_frame.columns = ["image1", "image2", "label"]
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__(self, index):
        image1_path = os.path.join(self.image_dir, self.data_frame.iloc[index, 0])
        image2_path = os.path.join(self.image_dir, self.data_frame.iloc[index, 1])

        img1 = Image.open(image1_path).convert("L")
        img2 = Image.open(image2_path).convert("L")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor([float(self.data_frame.iloc[index, 2])], dtype=torch.float32)
        return img1, img2, label

    def __len__(self):
        return len(self.data_frame)

# View sample images to ensure correct loading
def visualize_samples(dataset, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        img1, img2, label = dataset[i]
        axes[i].imshow(torch.cat((img1, img2), dim=2).permute(1, 2, 0).cpu().numpy(), cmap="gray")
        axes[i].set_title(f"Label: {int(label.item())}")
        axes[i].axis("off")
    plt.show()

# Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 2)
        )

    def forward_once(self, x):
        x = self.cnn1(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# Contrastive Loss with Euclidean Distance
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Compute Euclidean distance between output1 and output2
        euclidean_distance = F.pairwise_distance(output1, output2, p=2)
        
        # Contrastive loss formula using Euclidean distance
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + 
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# Train loop
def train(train_loader):
    net.train()
    running_loss = 0
    for img1, img2, label in tqdm(train_loader, desc="Training", leave=False):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        optimizer.zero_grad()
        output1, output2 = net(img1, img2)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)

# Evaluation loop
def evaluate(eval_loader):
    net.eval()
    running_loss = 0
    with torch.no_grad():
        for img1, img2, label in tqdm(eval_loader, desc="Evaluating", leave=False):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output1, output2 = net(img1, img2)
            loss = criterion(output1, output2, label)
            running_loss += loss.item()
    return running_loss / len(eval_loader)

# Metrics: ROC-AUC and PR-AUC
def compute_metrics(eval_loader):
    net.eval()
    distances, labels = [], []
    with torch.no_grad():
        for img1, img2, label in eval_loader:
            img1, img2 = img1.to(device), img2.to(device)
            output1, output2 = net(img1, img2)
            # Use Euclidean distance for evaluation
            dist = F.pairwise_distance(output1, output2, p=2).cpu().numpy()
            distances.extend(dist)
            labels.extend(label.cpu().numpy())
    roc_auc = roc_auc_score(labels, np.array(distances))
    precision, recall, _ = precision_recall_curve(labels, np.array(distances))
    pr_auc = auc(recall, precision)
    return roc_auc, pr_auc

# Main training script
if __name__ == "__main__":
    # Dataset and DataLoader
    train_dataset = SiameseDataset(config.training_csv, config.training_dir, transform=transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor()        
    ]))

    eval_dataset = SiameseDataset(config.testing_csv, config.testing_dir, transform=transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor()
    ]))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Visualize sample images
    print("Visualizing sample images...")
    visualize_samples(train_dataset)

    # Initialize model, loss, and optimizer
    net = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)

    # Define StepLR scheduler: Decay learning rate every 3 epochs by a factor of 0.5
    step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # Early stopping configuration
    early_stopping_patience = 6
    early_stopping_counter = 0

    best_eval_loss = 9999
    best_roc_auc = 0

    # Directory to save the model
    save_dir = "./model_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    for epoch in range(1, 21):  # 20 epochs
        print(f"\nEpoch [{epoch}/20]")

        # Training phase
        train_loss = train(train_loader)

        # Evaluation phase
        eval_loss = evaluate(eval_loader)
        roc_auc, pr_auc = compute_metrics(eval_loader)

        # Log results
        print(f"Train Loss: {train_loss} - Eval Loss: {eval_loss}")
        print(f"ROC-AUC: {roc_auc} - PR-AUC: {pr_auc}")

        # Step the scheduler to update the learning rate
        step_lr_scheduler.step()

        # Save the model if it improves
        if eval_loss < best_eval_loss or roc_auc > best_roc_auc:
            best_eval_loss = eval_loss
            best_roc_auc = roc_auc
            early_stopping_counter = 0
            print("Saving improved model...")
            torch.save(net.state_dict(), os.path.join(save_dir, "best_model.pth"))
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break

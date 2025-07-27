import os
import re
import torch
import wandb
import argparse
import skimage.io

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision.transforms import v2

# Select GPU, else MPS, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class OCTAData(Dataset):
    def __init__(self, images_path, masks_path, transform=None, margin_size=20):
        self.image_paths = []
        self.mask_paths = []
        self.transform = transform
        self.margin_size = margin_size

        image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        mask_files = set(os.listdir(masks_path))

        for image_file in image_files:
            img_path = os.path.join(images_path, image_file)
            image_number = os.path.splitext(image_file)[0].split("_")[-1]
            mask_pattern = re.compile(rf"Mask_{image_number}.tif")
            matched_mask = next((m for m in mask_files if mask_pattern.match(m)), None)

            if not matched_mask:
                print(f" Warning: No mask found for {image_file}. Skipping.")
                continue

            self.image_paths.append(img_path)
            self.mask_paths.append(os.path.join(masks_path, matched_mask))

    def __getitem__(self, idx):
        img = skimage.io.imread(self.image_paths[idx], as_gray=True)
        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img, dtype=torch.float32) / 255.0

        mask = skimage.io.imread(self.mask_paths[idx], as_gray=True)
        mask = (mask > 0.5).astype(np.uint8)  # ensure binary
        if self.margin_size > 0:
            mask = mask[self.margin_size:-self.margin_size, self.margin_size:-self.margin_size]
        mask = torch.tensor(mask, dtype=torch.int64)

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask

    def __len__(self):
        return len(self.image_paths)

class UNet128(torch.nn.Module):
    def __init__(self, out_channels=2, dropout=0.2):
        super().__init__()
        
        def double_conv(in_channels, out_channels):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout2d(p=dropout),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout2d(p=dropout),
            )

        self.down1 = double_conv(1, 32)
        self.pool1 = torch.nn.MaxPool2d(2)

        self.down2 = double_conv(32, 64)
        self.pool2 = torch.nn.MaxPool2d(2)

        self.down3 = double_conv(64, 128)
        self.pool3 = torch.nn.MaxPool2d(2)

        self.bottleneck = double_conv(128, 256)

        self.up3 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = double_conv(256, 128)

        self.up2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv2 = double_conv(128, 64)

        self.up1 = torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv1 = double_conv(64, 32)

        self.final_conv = torch.nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        l1 = self.down1(x)
        l2 = self.down2(self.pool1(l1))
        l3 = self.down3(self.pool2(l2))

        bottleneck = self.bottleneck(self.pool3(l3))

        up3 = self.up3(bottleneck)
        l3_resized = F.interpolate(l3, size=up3.shape[2:], mode='bilinear', align_corners=False)
        up3 = torch.cat([up3, l3_resized], dim=1)
        up3 = self.upconv3(up3)

        up2 = self.up2(up3)
        l2_resized = F.interpolate(l2, size=up2.shape[2:], mode='bilinear', align_corners=False)
        up2 = torch.cat([up2, l2_resized], dim=1)
        up2 = self.upconv2(up2)

        up1 = self.up1(up2)
        l1_resized = F.interpolate(l1, size=up1.shape[2:], mode='bilinear', align_corners=False)
        up1 = torch.cat([up1, l1_resized], dim=1)
        up1 = self.upconv1(up1)

        out = self.final_conv(F.interpolate(up1, size=x.shape[2:], mode='bilinear', align_corners=False))
        return out

class DiceCELoss(torch.nn.Module):
    def __init__(self, weight=None, dice_weight=0.5, ce_weight=0.5, epsilon=1e-6):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.epsilon = epsilon

    def forward(self, logits, labels):
        ce_loss = self.ce(logits, labels)

        probs = F.softmax(logits, dim=1)
        true_1_hot = F.one_hot(labels, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * true_1_hot, dims)
        cardinality = torch.sum(probs + true_1_hot, dims)
        dice_loss = 1. - (2. * intersection + self.epsilon) / (cardinality + self.epsilon)
        dice_loss = dice_loss.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss    

def weighted_accuracy(pred_mask, label_np, weight_foreground=0.7, weight_background=0.3):
    foreground_mask = (label_np == 1)
    background_mask = (label_np == 0)

    foreground_correct = (pred_mask == label_np) & foreground_mask
    background_correct = (pred_mask == label_np) & background_mask

    foreground_acc = foreground_correct.sum(axis=(1, 2)) / (foreground_mask.sum(axis=(1, 2)) + 1e-10)
    background_acc = background_correct.sum(axis=(1, 2)) / (background_mask.sum(axis=(1, 2)) + 1e-10)

    accuracy_batch = (weight_foreground * foreground_acc) + (weight_background * background_acc)
    return accuracy_batch


def dice_score(pred_mask, label_mask, epsilon=1e-6):
    pred_mask = (pred_mask == 1)
    label_mask = (label_mask == 1)

    intersection = (pred_mask & label_mask).sum(axis=(1, 2))
    union = pred_mask.sum(axis=(1, 2)) + label_mask.sum(axis=(1, 2))

    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice


def trainUNet128(trainData, valData=None, model=None, lr=0.0001, nr_epochs=100, n_train_batch=10, patience=10, min_delta=1e-4, dataset_size=None):

    wandb.init(
        project="OCTA_Segmentation_dataset_new",
        config={
            "learning_rate": lr,
            "epochs": nr_epochs,
            "batch_size": n_train_batch,
            "dataset_size": dataset_size,
            "patience": patience,
            "min_delta": min_delta,
            "architecture": "UNet128",
            "device": str(device),
        }
    )

    if model is None:
        model = UNet128().to(device)
    wandb.watch(model, log="all", log_freq=10)

    trainloader = torch.utils.data.DataLoader(trainData, batch_size=n_train_batch, shuffle=True, drop_last=True)
    valloader = torch.utils.data.DataLoader(valData, batch_size=n_train_batch) if valData else None

    loss_function = DiceCELoss(dice_weight=0.7, ce_weight=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []
    val_losses = []
    batch_losses = []
    accuracy_scores = []
    best_model_state = None
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(nr_epochs):
        model.train()
        epoch_loss = 0.0
        accuracy_scores = []
        dice_scores = []

        for batch in trainloader:
            image_batch, label_batch = batch
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            if label_batch.ndim == 3:
                label_batch = label_batch.unsqueeze(1)
            label_batch = F.interpolate(label_batch.float(), size=image_batch.shape[2:], mode='nearest').squeeze(1).long()

            optimizer.zero_grad()
            logits_batch = model(image_batch)
            loss = loss_function(logits_batch, label_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_losses.append(loss.item())

            pred_mask = logits_batch.argmax(dim=1).cpu().numpy()
            label_np = label_batch.cpu().numpy()
            acc_batch = weighted_accuracy(pred_mask, label_np)
            dice_batch = dice_score(pred_mask, label_np)
            accuracy_scores.extend(acc_batch.tolist())
            dice_scores.extend(dice_batch.tolist())

        avg_epoch_loss = epoch_loss / len(trainloader)
        epoch_losses.append(avg_epoch_loss)

        if valloader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_img, val_mask in valloader:
                    val_img, val_mask = val_img.to(device), val_mask.to(device)
                    if val_mask.ndim == 3:
                        val_mask = val_mask.unsqueeze(1)
                    val_mask = F.interpolate(val_mask.float(), size=val_img.shape[2:], mode='nearest').squeeze(1).long()
                    val_logits = model(val_img)
                    loss = loss_function(val_logits, val_mask)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(valloader)
            val_losses.append(avg_val_loss)
            print(f'\rEpoch {epoch}/{nr_epochs}, Train Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}', end='')

            wandb.log({
                "epoch": epoch,
                "train_loss": avg_epoch_loss,
                "val_loss": avg_val_loss,
                "mean_weighted_accuracy": np.mean(accuracy_scores),
                "mean_dice_score": np.mean(dice_scores)
            })

            if avg_val_loss + min_delta < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}.")
                    break
        else:
            print(f'\rEpoch {epoch}/{nr_epochs}, Train Loss: {avg_epoch_loss:.4f}', end='')
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_epoch_loss,
                "mean_weighted_accuracy": np.mean(accuracy_scores),
                "mean_dice_score": np.mean(dice_scores)
            })

    if best_model_state:
        model.load_state_dict(best_model_state)

    wandb.finish()
    return model, best_model_state, epoch_losses, batch_losses, val_losses, accuracy_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--suffix', type=str, default="")
    parser.add_argument('--dataset_size', type=int, default=None)
    args = parser.parse_args()


    images_path = r"/zhome/5a/9/203276/Documents/MONAI_thesis/Segmentation/Segmentation_Dataset/cleared_images"
    masks_path = r"/zhome/5a/9/203276/Documents/MONAI_thesis/Segmentation/Segmentation_Dataset/new_masks"

    aug_transforms = v2.Compose([
        v2.GaussianBlur(kernel_size=(1, 1), sigma=(0.1, 0.2)),
    ])

    dataset = OCTAData(images_path, masks_path, transform=aug_transforms)

    # Apply dataset size limit
    if args.dataset_size is not None and args.dataset_size < len(dataset):
        dataset, _ = torch.utils.data.random_split(dataset, [args.dataset_size, len(dataset) - args.dataset_size])
        
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    TrainData, ValData, TestData = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    model, best_model_state, epoch_losses, batch_losses, val_losses, accuracy_scores = trainUNet128(
        TrainData,
        valData=ValData,
        lr=args.lr,
        nr_epochs=args.epochs,
        n_train_batch=args.batch_size,
        patience=args.patience,
        dataset_size=args.dataset_size
    )

    model_filename = f"model_mark7_{args.suffix}.pt"
    save_path = os.path.join("/zhome/5a/9/203276/Documents/MONAI_thesis/Segmentation/Segmentation_Models", model_filename)
    torch.save(best_model_state, save_path)


if __name__ == "__main__":
    main()
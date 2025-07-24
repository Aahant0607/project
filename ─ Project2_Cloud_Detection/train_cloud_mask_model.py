This script outlines a complete pipeline for a cloud segmentation task using Sentinel-2 satellite imagery from the `CloudSEN12` dataset. It leverages powerful libraries like `PyTorch`, `segmentation-models-pytorch` for the model architecture, and `albumentations` for data augmentation. The process includes downloading and localizing data, defining a custom dataset and data loaders, training a U-Net model with an EfficientNet backbone, and finally, evaluating and visualizing the results.

### **1. Setup: Install & Import Libraries**

First, we install the necessary libraries. `segmentation-models-pytorch` provides pre-trained segmentation models, `albumentations` is used for efficient data augmentation, and `tacoreader` allows us to stream the dataset.

```python
# ==================================================================================
# 1. SETUP: INSTALL & IMPORT LIBRARIES
# ==================================================================================
!pip install -q segmentation-models-pytorch albumentations tacoreader rasterio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

import tacoreader
import rasterio as rio
from rasterio.errors import RasterioIOError
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import os
import time
from pathlib import Path
import random
```

-----

### **2. Configuration**

A `Config` class is used to store all hyperparameters and settings in one place. This makes it easy to experiment with different backbones, image sizes, learning rates, etc. We're optimizing for speed by using a reasonable `BATCH_SIZE`, `NUM_WORKERS`, and a powerful `efficientnet-b2` backbone.

```python
# ==================================================================================
# 2. CONFIGURATION
# ==================================================================================
class Config:
    """Configuration class for hyperparameters and settings."""
    DATASET_NAME = "tacofoundation:cloudsen12-l1c"
    IMG_SIZE = 256
    BACKBONE = "efficientnet-b2"
    BATCH_SIZE = 32
    NUM_WORKERS = 2  # Number of parallel workers for data loading
    BANDS = [4, 3, 2]  # RGB bands from Sentinel-2
    VAL_SPLIT = 0.2
    MODEL_ARCH = "Unet"
    MAX_EPOCHS = 50
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WEIGHT_DECAY = 1e-2
    EARLY_STOPPING_PATIENCE = 5
```

-----

### **3. Data Localization**

To accelerate training, we first download a subset of the dataset from the remote source and save it to local storage. This `localize_dataset` function handles the download process, reads the specified image bands and masks, and saves them as NumPy (`.npy`) files. It includes retry logic to handle potential network issues during the initial connection.

```python
# ==================================================================================
# 3. DATA LOCALIZATION
# ==================================================================================
def localize_dataset(cfg, num_samples=3000):
    """Downloads and saves dataset samples locally for faster access."""
    print("--- Step 3: Localizing dataset ---")
    
    LOCAL_IMG_DIR = Path("./temp_data/images")
    LOCAL_MASK_DIR = Path("./temp_data/masks")
    LOCAL_IMG_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_MASK_DIR.mkdir(parents=True, exist_ok=True)

    # Retry loop to handle network errors during the initial dataset load
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Loading remote dataset reference (Attempt {attempt + 1}/{max_retries})...")
            tacoreader_dataset = tacoreader.load(cfg.DATASET_NAME)
            print("✅ Remote reference loaded.")
            break 
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Failed with error: {e}. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("Fatal: Could not load dataset after multiple retries.")
                raise e

    all_indices = list(range(len(tacoreader_dataset)))
    demo_indices = all_indices[:num_samples]

    print(f"Downloading {len(demo_indices)} samples to local storage...")
    for idx in tqdm(demo_indices, desc="Downloading Samples"):
        try:
            s2_l1c_path = tacoreader_dataset.read(idx).read(0)
            s2_label_path = tacoreader_dataset.read(idx).read(1)

            with rio.open(s2_l1c_path) as src:
                image = src.read(cfg.BANDS)
            with rio.open(s2_label_path) as src:
                mask = src.read(1)
            
            np.save(LOCAL_IMG_DIR / f"{idx}.npy", image)
            np.save(LOCAL_MASK_DIR / f"{idx}.npy", mask)
        except Exception as e:
            print(f"Warning: Could not download sample {idx}. Skipping. Error: {e}")
            
    print("✅ Dataset localization complete.")
    return demo_indices, LOCAL_IMG_DIR, LOCAL_MASK_DIR
```

-----

### **4. Custom PyTorch Dataset**

The `LocalCloudDataset` class is a custom `torch.utils.data.Dataset`. It's designed to read the `.npy` files we saved locally. In its `__getitem__` method, it loads an image and its corresponding mask, performs necessary preprocessing (like scaling pixel values and converting the mask to a binary format), and applies the specified data augmentations.

```python
# ==================================================================================
# 4. PYTORCH DATASET FOR LOCAL FILES
# ==================================================================================
class LocalCloudDataset(Dataset):
    """Custom PyTorch Dataset to load images and masks from local .npy files."""
    def __init__(self, sample_indices, img_dir, mask_dir, transforms):
        # Ensure we only use indices for which files were successfully downloaded
        self.sample_indices = [idx for idx in sample_indices if (img_dir / f"{idx}.npy").exists()]
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]
        
        # Load image and transpose to (H, W, C) for albumentations
        image_uint16 = np.load(self.img_dir / f"{sample_idx}.npy").transpose(1, 2, 0)
        mask = np.load(self.mask_dir / f"{sample_idx}.npy")

        # Normalize image: clip high values and scale to 0-255 uint8
        max_val = 4000
        image_scaled = np.clip(image_uint16, 0, max_val) / max_val
        image = (image_scaled * 255).astype(np.uint8)
        
        # Binarize mask: 'thin' (1) and 'thick' (2) clouds are treated as cloud (1.0)
        binary_mask = np.where((mask == 1) | (mask == 2), 1.0, 0.0).astype(np.float32)
        
        # Apply augmentations
        augmented = self.transforms(image=image, mask=binary_mask)
        return augmented['image'], augmented['mask'].unsqueeze(0) # Add channel dim to mask
```

-----

### **5. Augmentations, Loss, and Training Loops**

This section defines the core components for training:

  * **`get_transforms`**: Creates augmentation pipelines for training (with flips, rotations, color jitter) and validation (only resizing and normalization).
  * **`SegmentationLoss`**: A composite loss function combining Focal Loss and Dice Loss. This is effective for imbalanced datasets, where the object of interest (clouds) might not cover the whole image. The loss is a weighted sum: $L = 0.25 \\times L\_{Focal} + 0.75 \\times L\_{Dice}$.
  * **`train_one_epoch` & `evaluate`**: Standard training and evaluation functions. They use automatic mixed precision (`autocast` and `GradScaler`) to speed up training on compatible GPUs.

<!-- end list -->

```python
# ==================================================================================
# 5. AUGMENTATIONS, LOSS, AND TRAINING/EVALUATION UTILITIES
# ==================================================================================
def get_transforms(cfg, is_train=True):
    """Returns an albumentations augmentation pipeline."""
    if is_train:
        return A.Compose([
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
            A.HorizontalFlip(p=0.5), 
            A.VerticalFlip(p=0.5), 
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.75),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])

class SegmentationLoss(nn.Module):
    """Composite loss combining Focal Loss and Dice Loss."""
    def __init__(self):
        super().__init__()
        self.focal = smp.losses.FocalLoss(mode='binary')
        self.dice = smp.losses.DiceLoss(mode='binary')
    def forward(self, y_pred, y_true):
        return 0.25 * self.focal(y_pred, y_true) + 0.75 * self.dice(y_pred, y_true)

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        # Use mixed precision
        with autocast():
            preds = model(images)
            loss = loss_fn(preds, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, loss_fn, device):
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss = 0.0
    all_preds, all_masks = [], []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating", leave=False):
            images, masks = images.to(device), masks.to(device)
            with autocast():
                preds = model(images)
                loss = loss_fn(preds, masks)
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(preds))
            all_masks.append(masks)
            
    if not all_preds: return 0.0, 0.0
    
    all_preds = torch.cat(all_preds).cpu()
    all_masks = torch.cat(all_masks).cpu()
    
    # Calculate IoU score for the epoch
    tp, fp, fn, tn = smp.metrics.get_stats((all_preds > 0.5).long(), all_masks.long(), mode='binary')
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
    
    return total_loss / len(loader), iou_score
```

-----

### **6. Main Execution: Training & Evaluation**

This is the main block where the script is executed. It performs the following steps:

1.  **Instantiates** the `Config`.
2.  **Localizes the data** (or skips if already done). A check is included to delete old model checkpoints to prevent errors if the architecture changes.
3.  **Splits** the data indices into training and validation sets.
4.  **Creates** `Dataset` and `DataLoader` instances for both sets.
5.  **Initializes** the `smp.Unet` model, the custom loss function, optimizer (`AdamW`), and a learning rate scheduler.
6.  **Runs the training loop**, saving the model with the best validation Intersection over Union (IoU) score. Early stopping is implemented to halt training if the IoU doesn't improve for a set number of epochs.
7.  After training, it **loads the best model** and performs a final, detailed evaluation.
8.  Finally, it **visualizes** predictions on a few random validation samples to provide a qualitative assessment of the model's performance.

<!-- end list -->

```python
# ==================================================================================
# 6. MAIN EXECUTION BLOCK
# ==================================================================================
if __name__ == '__main__':
    cfg = Config()
    
    # STEP 1: Run the localization process.
    # To prevent re-downloading on subsequent runs, you can comment this line out
    # and use the logic below to load from existing directories.
    # demo_indices, img_dir, mask_dir = localize_dataset(cfg)
    
    # --- Logic to use already-downloaded data ---
    # This is useful in environments like Kaggle where data persists between sessions.
    model_path = Path('best_model.pth')
    if model_path.exists():
        print(f"🗑️ Deleting old model checkpoint to start fresh: {model_path}")
        model_path.unlink()
    
    print("--- Bypassing download, assuming data exists locally ---")
    # Define the path where data was saved. Update if necessary.
    img_dir = Path("./temp_data/images")
    mask_dir = Path("./temp_data/masks")
    
    # Recreate the list of sample indices from the files present on disk
    if not img_dir.exists():
        print(f"Error: Local data directory not found at {img_dir}. Please run localize_dataset() first.")
    else:
        demo_indices = [int(p.stem) for p in img_dir.glob("*.npy")]
        print(f"✅ Found {len(demo_indices)} local samples at {img_dir}")
        
        # STEP 2: SPLIT DATA & CREATE DATALOADERS
        print("\n--- Step 2: Preparing data loaders ---")
        train_indices, val_indices = train_test_split(demo_indices, test_size=cfg.VAL_SPLIT, random_state=42)

        train_dataset = LocalCloudDataset(train_indices, img_dir, mask_dir, get_transforms(cfg, is_train=True))
        val_dataset = LocalCloudDataset(val_indices, img_dir, mask_dir, get_transforms(cfg, is_train=False))

        train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

        # STEP 3: INITIALIZE MODEL & TRAINING COMPONENTS
        print(f"\n--- Step 3: Initializing {cfg.MODEL_ARCH} with {cfg.BACKBONE} backbone ---")
        model = smp.Unet(encoder_name=cfg.BACKBONE, encoder_weights="imagenet", in_channels=3, classes=1).to(cfg.DEVICE)
        loss_fn = SegmentationLoss().to(cfg.DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        scaler = GradScaler()
        
        # STEP 4: TRAINING LOOP
        print(f"\n--- Step 4: Starting Training on {cfg.DEVICE.upper()} ---")
        best_iou, patience_counter = 0.0, 0
        for epoch in range(cfg.MAX_EPOCHS):
            print(f"--- Epoch {epoch+1}/{cfg.MAX_EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f} ---")
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, cfg.DEVICE, scaler)
            val_loss, val_iou = evaluate(model, val_loader, loss_fn, cfg.DEVICE)
            scheduler.step(val_loss)
            
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
            
            if val_iou > best_iou:
                best_iou, patience_counter = val_iou, 0
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"  ✨ New best model saved with IoU: {best_iou:.4f}")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{cfg.EARLY_STOPPING_PATIENCE}")
            
            if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
                print(f"\nStopping early after {patience_counter} epochs with no improvement.")
                break
                
        print(f"\n✅ Training finished! Best validation IoU: {best_iou:.4f}")

        # ==================================================================================
        # 7. FINAL EVALUATION & VISUALIZATION
        # ==================================================================================
        print("\n--- Step 7: Final Evaluation and Visualization ---")

        # Load the best performing model's weights
        model.load_state_dict(torch.load('best_model.pth', map_location=cfg.DEVICE))

        def evaluate_all_metrics(model, loader, device):
            """Calculates IoU, F1, and Accuracy on a given data loader."""
            model.eval()
            all_preds, all_masks = [], []
            with torch.no_grad():
                for images, masks in tqdm(loader, desc="Final Evaluation", leave=False):
                    images, masks = images.to(device), masks.to(device)
                    with autocast():
                        preds = model(images)
                    all_preds.append(torch.sigmoid(preds))
                    all_masks.append(masks)

            all_preds = torch.cat(all_preds).cpu()
            all_masks = torch.cat(all_masks).cpu()
            
            # CRITICAL FIX: Apply a 0.5 threshold to predictions BEFORE calculating stats
            preds_binary = (all_preds > 0.5).long()
            masks_long = all_masks.long()
            
            tp, fp, fn, tn = smp.metrics.get_stats(preds_binary, masks_long, mode='binary')
            
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro')
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction='micro')
            
            return iou_score, f1_score, accuracy

        # Run the final evaluation
        val_iou, val_f1, val_acc = evaluate_all_metrics(model, val_loader, cfg.DEVICE)
        print("\n--- Final Model Performance on Validation Set ---")
        print(f"📊 IoU Score: {val_iou:.4f}")
        print(f"📊 F1 Score: {val_f1:.4f}")
        print(f"📊 Pixel Accuracy: {val_acc:.4f}")
        print("-------------------------------------------------")

        # Visualize predictions for 5 random samples
        print("\nGenerating visual results for 5 random samples...")
        model.eval()
        num_samples_to_show = 5
        sample_indices_to_show = random.sample(range(len(val_dataset)), num_samples_to_show)

        def denormalize(tensor):
            """Denormalizes a tensor for plotting."""
            tensor = tensor.clone().permute(1, 2, 0)
            tensor.mul_(torch.tensor([0.5, 0.5, 0.5])).add_(torch.tensor([0.5, 0.5, 0.5]))
            return torch.clamp(tensor, 0, 1)

        for i in sample_indices_to_show:
            image_tensor, true_mask = val_dataset[i]
            
            with torch.no_grad():
                image_tensor_batch = image_tensor.unsqueeze(0).to(cfg.DEVICE)
                pred_mask = model(image_tensor_batch)
                pred_mask = (torch.sigmoid(pred_mask) > 0.5).squeeze(0).cpu()

            image_to_plot = denormalize(image_tensor)
            true_mask_to_plot = true_mask.squeeze()
            pred_mask_to_plot = pred_mask.squeeze()

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            ax1.imshow(image_to_plot)
            ax1.set_title("Original Image")
            ax1.axis('off')

            ax2.imshow(true_mask_to_plot, cmap='gray')
            ax2.set_title("True Mask (Ground Truth)")
            ax2.axis('off')

            ax3.imshow(pred_mask_to_plot, cmap='gray')
            ax3.set_title("Predicted Mask")
            ax3.axis('off')

            plt.tight_layout()
            plt.show()

        print("\n✅ Visualization finished.")

```

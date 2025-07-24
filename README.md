### **Team Members**
* **Name:**     Aahant Kumar
* **Email:**    royaahantbhardwaj@gmail.com
* **Roll No.:** 24/A11/001 
* **Name:**     Divyansh Sharma
* **Email:**    divyanshsharma_23ch023@dtu.ac.in
* **Roll No.:** 23/CH/023


Task 1

# SAR-to-EO Translation with CycleGAN  
**Using Sentinel-1 (SAR) to Sentinel-2 (EO) Winter Scenes**  

## Overview  
This project implements and trains a **CycleGAN** model to translate **Sentinel-1 SAR** images into **Sentinel-2 EO** images using the **Sen12MS** dataset.  
We experiment with multiple **band configurations** of Sentinel-2 outputs, evaluate the model using **spectral metrics**, and explore potential architectural improvements for enhanced performance.  

Our work is based on the original **CycleGAN (Zhu et al., 2017)** architecture, extended for **multi-channel remote sensing data** using **ResNet-based generators** and **PatchGAN discriminators**.  

---

## Dataset  
We use the **Sen12MS Winter subset**:  

- **SAR**: `ROIs2017_winter_s1.tar.gz`  
- **EO**: `ROIs2017_winter_s2.tar.gz`  
- [Download here](https://dataserv.ub.tum.de/s/m1474000)  

Each EO image includes **13 spectral bands** (Sentinel-2 L2A product).  

---

## Preprocessing  
- Extract `.tar.gz` archives and organize into paired SAR-EO folders.  
- Select winter scenes for training.  
- **Band Selection** for different experiments:  
  - **RGB**: B4 (Red), B3 (Green), B2 (Blue)  
  - **NIR + SWIR + Red Edge**: B8 (NIR), B11 (SWIR), B5 (Red Edge)  
  - **RGB + NIR**: B4, B3, B2, B8  
- Resize to **256×256 tiles**.  
- Normalize all inputs to **[-1, 1]**.  

---

## Model  
**Architecture**:  
- **Generator**: ResNet-based, 9 residual blocks (for 256×256 inputs).  
- **Discriminator**: 70×70 PatchGAN.  

**Loss Functions**:  
- Adversarial Loss (LSGAN)  
- Cycle Consistency Loss (λ = 10)  
- Identity Loss (optional)  

**Hyperparameters:**  
- Learning Rate: `2e-4`  
- Epochs: `20` (on 50% dataset; scalable to full dataset)  
- Optimizer: Adam (`β1 = 0.5, β2 = 0.999`)  

---

## Experiments & Results  

### 1. SAR → EO (RGB)  
- **Bands**: B4, B3, B2  
- **PSNR**: 24 dB  
- **SSIM**: 0.89  
- **Observation**: Model captures realistic colors & textures, though discriminator occasionally fooled. Full dataset training expected to improve up to **28 dB PSNR**.  

### 2. SAR → EO (NIR + SWIR + Red Edge)  
- **Bands**: B8, B11, B5  
- **PSNR**: 19 dB  
- **SSIM**: 0.40  
- **Observation**: Captures broad structures, but fine spectral details are limited.  

### 3. SAR → EO (RGB + NIR)  
- **Bands**: B4, B3, B2, B8  
- **PSNR**: 22.02 dB  
- **SSIM**: 0.1706  
- **Observation**: Good for structural information (edges, shapes), limited spectral fidelity.  

---

## Postprocessing  
- **Denormalize** outputs back to `[0, 1]`.  
- **Visualize** using false-color composites and histograms.  
- **Compute Vegetation Indices** (e.g., NDVI) for validation.  

---

## Performance Metrics  
- **PSNR** (Peak Signal-to-Noise Ratio)  
- **SSIM** (Structural Similarity Index)  
- **Spectral Metrics**:  
  - NDVI consistency  
  - Band-wise RMSE  

---

## Challenges & Observations  
- Discriminator often **gets fooled**, particularly for non-RGB bands.  
- Spectral channels (**SWIR, Red Edge**) are harder to reconstruct due to **low correlation with SAR**.  
- **RGB bands** show best structural & visual quality.  
- Training on **full dataset + tuned hyperparameters** expected to improve results significantly.  

---

## Future Work  
- Train on **full Sen12MS dataset** with longer schedules (≥100 epochs).  
- **UNet-based generators** for better spatial fidelity.  
- **Multi-frequency CycleGAN** (Hyper-CycleGAN, 2024) for enhanced spectral feature transfer.  
- **Multi-task losses** (e.g., perceptual loss, spectral regularization).  

---

## References  
1. Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017.  
2. Schmitt et al., *Sen12MS – A Curated Dataset of Georeferenced Multi-Spectral Sentinel-1/2 Imagery for Deep Learning and Data Fusion*, ISPRS 2019.  
3. *Hyper-CycleGAN: Multi-Frequency CycleGAN for Remote Sensing* (2024).  




Task 2
# Cloud Segmentation with U-Net and Sentinel-2 Imagery

This project implements a complete deep learning pipeline for semantic segmentation of clouds in Sentinel-2 satellite images. It utilizes the `CloudSEN12` dataset, a U-Net model with an EfficientNet backbone, and the PyTorch ecosystem.


***
### **b. Project Overview**
The goal of this project is to accurately identify and segment cloud cover from satellite imagery. This is a crucial preprocessing step for many remote sensing applications, as clouds can obscure ground features.

This repository provides an end-to-end solution that:
1.  Downloads and prepares a subset of the `CloudSEN12` dataset.
2.  Preprocesses the image data and masks for training.
3.  Implements robust data augmentation to improve model generalization.
4.  Trains a **U-Net** model with a pre-trained **EfficientNet-B2** backbone.
5.  Evaluates the model using Intersection over Union (IoU), F1-Score, and Pixel Accuracy.
6.  Visualizes the model's predictions against the ground truth masks.

***
### **c. Instructions to Run Code**

1.  **Clone the Repository (Optional):**
    If this were a git repository, you would clone it first.
    ```bash
    git clone [https://your-repository-url.com/cloud-segmentation.git](https://your-repository-url.com/cloud-segmentation.git)
    cd cloud-segmentation
    ```

2.  **Install Dependencies:**
    Install all the required Python libraries using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Script:**
    Execute the main Python script named `train_cloud_mask_model.py`.

    ```bash
    python train.py
    ```
    * **First Run:** The script will automatically download ~3000 samples from the `CloudSEN12` dataset and save them to a local `./temp_data/` directory. This may take some time depending on your network connection.
    * keep this piece of code commented:
    demo_indices, img_dir, mask_dir = localize_dataset(cfg)
    
    * keep this piece of code uncommented:
    print("--- Bypassing download, assuming data exists locally ---")
    img_dir = Path("./temp_data/images")
    mask_dir = Path("./temp_data/masks")
    if not img_dir.exists():
        print(f"Error: Local data directory not found at {img_dir}. Please run localize_dataset() first.")
    else:
        demo_indices = [int(p.stem) for p in img_dir.glob("*.npy")]
        print(f"✅ Found {len(demo_indices)} local samples at {img_dir}")

   
    * **Subsequent Runs:** The script is designed to detect the existing local data and will skip the download process, proceeding directly to training.
    * keep this piece of code uncommented:
    demo_indices, img_dir, mask_dir = localize_dataset(cfg)
    
    * keep this piece of code commented:
    print("--- Bypassing download, assuming data exists locally ---")
    img_dir = Path("./temp_data/images")
    mask_dir = Path("./temp_data/masks")
    if not img_dir.exists():
        print(f"Error: Local data directory not found at {img_dir}. Please run localize_dataset() first.")
    else:
        demo_indices = [int(p.stem) for p in img_dir.glob("*.npy")]
        print(f"✅ Found {len(demo_indices)} local samples at {img_dir}")

***
### **d. Description**

#### **i. Data Preprocessing Steps**

The data pipeline is designed for efficiency and correctness:
* **Data Source:** The project uses the `tacofoundation:cloudsen12-l1c` dataset, accessed via the `tacoreader` library.
* **Localization:** To accelerate I/O operations during training, a subset of 3000 image/mask pairs is downloaded and saved locally as NumPy (`.npy`) files.
* **Band Selection:** Only the true-color **RGB bands (4, 3, 2)** are used from the Sentinel-2 L1C data.
* **Image Normalization:** The raw 16-bit image data is clipped at a maximum pixel value of 4000 (to handle sensor saturation) and then scaled to a standard 8-bit format (0-255) suitable for the model's pre-trained backbone.
* **Mask Binarization:** The original ground truth masks have multiple classes. For this binary segmentation task, the 'thin cloud' (value 1) and 'thick cloud' (value 2) labels are merged into a single 'cloud' class (1.0), with all other pixels set to 'no cloud' (0.0).
* **Data Augmentation:** The `albumentations` library is used to create robust training data. Augmentations include:
    * Resizing to $256 \times 256$ pixels.
    * Geometric transforms: Horizontal Flips, Vertical Flips, and 90-degree Rotations.
    * Photometric transforms: Color Jitter (adjusting brightness, contrast, and saturation).

#### **ii. Models Used**
* **Architecture:** The core model is a **U-Net**, a convolutional neural network architecture widely used for biomedical and satellite image segmentation. It is sourced from the highly-regarded `segmentation-models-pytorch` library.
* **Backbone:** An **EfficientNet-B2** encoder, pre-trained on ImageNet, is used as the feature extractor (the "backbone") for the U-Net. This leverages transfer learning to achieve better performance with less training time.
* **Loss Function:** A composite loss function is employed to handle the potential class imbalance between cloud and non-cloud pixels. It is a weighted sum of **Focal Loss** and **Dice Loss**:
    $$ L = 0.25 \times L_{Focal} + 0.75 \times L_{Dice} $$
    This combination helps the model focus on hard-to-classify pixels while also directly optimizing for segmentation overlap (IoU).

#### **iii. Key Findings or Observations**
The model was trained until the validation IoU score stopped improving for 5 consecutive epochs (early stopping). The best model was saved and evaluated.

* **Quantitative Results:** The final model achieved excellent performance on the held-out validation set:
    * **IoU Score:** 0.7619
    * **F1 Score:** 0.8648
    * **Pixel Accuracy:** 0.9015

* **Qualitative Results:** The visual comparison of predicted masks against the ground truth shows that the model is highly effective. It successfully identifies the complex shapes and boundaries of clouds, with only minor discrepancies around the very fine edges.

* **Training Strategy:** The use of a pre-trained backbone, a composite loss function, and extensive data augmentation proved to be a successful strategy for this task. The `AdamW` optimizer and `ReduceLROnPlateau` scheduler provided stable convergence.

***
### **e. Tools and Frameworks**
* **Primary Framework:** **PyTorch**
* **Key Libraries:**
    * `segmentation-models-pytorch`: For pre-built U-Net model architecture.
    * `albumentations`: For high-performance data augmentation.
    * `tacoreader`: For streaming the CloudSEN12 dataset.
    * `rasterio`: For reading geospatial raster data.
    * `scikit-learn`: For splitting data into training and validation sets.
    * `numpy`: For numerical operations.
    * `matplotlib`: For visualization.

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

---


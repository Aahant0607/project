
---

# SAR to EO Dataset Preprocessing

This script performs preprocessing of Synthetic Aperture Radar (SAR) and Electro-Optical (EO) imagery datasets for machine learning tasks such as image translation, denoising, or classification. It reads `.tif` files from region folders, applies denoising, resizing, and normalization, and saves the processed tensors as `.pt` files.

---

## 📂 Folder Structure

Expected input folder structure:

```
ROIs2017_winter_s1/
└── ROIs2017_winter/
    ├── s1_ROI_1/
    │   ├── image1.tif
    │   └── image2.tif
    └── s1_ROI_2/
        └── ...
        
ROIs2017_winter_s2/
└── ROIs2017_winter/
    ├── s2_ROI_1/
    │   ├── image1.tif
    │   └── image2.tif
    └── s2_ROI_2/
        └── ...
```

After preprocessing, the output will be structured as:

```
output/
├── s1_ROI_1/
│   ├── SAR.pt
│   └── EO.pt
└── s1_ROI_2/
    ├── SAR.pt
    └── EO.pt
```

---

## ✅ Features

* Supports SAR (S1) and EO (S2) image modalities
* Applies **Non-Local Means Denoising** to reduce noise
* Resizes images to **224×224**
* Normalizes pixel values to **\[0, 1]**
* Outputs tensors stacked by time or channels using **PyTorch**

---

## ⚙️ Dependencies

* Python 3.7+
* PyTorch
* torchvision
* rasterio
* scikit-image
* numpy

Install required packages:

```bash
pip install torch torchvision rasterio scikit-image numpy
```

---

## 🚀 Usage

Run the script from the command line:

```bash
python preprocess.py
```

By default, it uses:

```python
s1_folder = r"ROIs2017_winter_s1\ROIs2017_winter"
s2_folder = r"ROIs2017_winter_s2\ROIs2017_winter"
output_folder = "output"
```

You can modify these paths directly in the script or adapt it to take command-line arguments.

---

## 🧠 How It Works

* **`read_and_process_images_torch()`**: Reads `.tif` files, denoises, resizes to 224×224, normalizes to \[0, 1], and stacks them into a tensor.
* **`process_all_torch()`**: Iterates over each ROI in the SAR folder, finds the corresponding EO folder, processes both, and saves them as `SAR.pt` and `EO.pt`.

---

## 📌 Notes

* Assumes ROI folder names follow the pattern `s1_` for SAR and `s2_` for EO, e.g., `s1_ROI_1` and `s2_ROI_1`.
* If an EO folder is missing, it will continue with SAR-only processing.
* Non-Local Means denoising uses `scikit-image` with estimated noise level.

---

## 📁 Output

Each processed ROI will contain:

* `SAR.pt`: Denoised, resized, normalized tensor of SAR images
* `EO.pt`: Denoised, resized, normalized tensor of EO images

These are saved in PyTorch format using `torch.save`.

---
By Aahant Kumar(24/A11/001) and Divyansh Sharma(23/CH/023) 

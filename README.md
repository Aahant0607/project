### **Team Members**
* **Name:**     Aahant Kumar
* **Email:**    royaahantbhardwaj@gmail.com
* **Roll No.:** 24/A11/001 
* **Name:**     Divyansh Sharma
* **Email:**    divyanshsharma_23ch023@dtu.ac.in
* **Roll No.:** 23/CH/023


Task 1



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
    * keep this piece of code uncommented:
    * keep this piece of code commented:
    * **Subsequent Runs:** The script is designed to detect the existing local data and will skip the download process, proceeding directly to training.
    * keep this piece of code uncommented:
    * keep this piece of code commented:

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

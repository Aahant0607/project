import os
import torch
import torchvision.transforms.functional as TF
import rasterio
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma

def normalize_255_torch(tensor):
    return tensor / 255.0

def resize_torch(img_tensor, size=(224, 224)):
    img_tensor = img_tensor.unsqueeze(0)  # Add channel dim: (1, H, W)
    img_resized = TF.resize(img_tensor, size, interpolation=TF.InterpolationMode.BILINEAR)
    return img_resized.squeeze(0)  # Back to (H, W)

def denoise_nlm_torch(img_tensor, h_factor=1.15, patch_size=5, patch_distance=6, fast_mode=True):
    img_np = img_tensor.cpu().numpy()
    sigma_est = np.mean(estimate_sigma(img_np, channel_axis=None))
    denoised = denoise_nl_means(
        img_np,
        h=h_factor * sigma_est,
        fast_mode=fast_mode,
        patch_size=patch_size,
        patch_distance=patch_distance,
        channel_axis=None
    )
    return torch.from_numpy(denoised).to(img_tensor.device).float()

def read_and_process_images_torch(folder_path, apply_denoise=True):
    imgs = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.tif'):
            with rasterio.open(os.path.join(folder_path, file)) as src:
                img = src.read(1).astype(np.float32)
                img_tensor = torch.from_numpy(img)

                if apply_denoise:
                    img_tensor = denoise_nlm_torch(img_tensor)

                img_tensor = resize_torch(img_tensor, (224, 224))
                img_tensor = normalize_255_torch(img_tensor)
                imgs.append(img_tensor)

    return torch.stack(imgs, dim=0) if imgs else None

def process_all_torch(s1_folder, s2_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    rois = [f for f in os.listdir(s1_folder) if os.path.isdir(os.path.join(s1_folder, f))]
    print(f"Found {len(rois)} ROIs.")

    for roi in rois:
        print(f"Processing ROI: {roi}")
        s1_roi_path = os.path.join(s1_folder, roi)
        s2_roi_name = roi.replace('s1_', 's2_', 1)
        s2_roi_path = os.path.join(s2_folder, s2_roi_name)
        out_roi_path = os.path.join(output_folder, roi)
        os.makedirs(out_roi_path, exist_ok=True)

        # EO
        if os.path.exists(s2_roi_path):
            eo_stack = read_and_process_images_torch(s2_roi_path, apply_denoise=True)
            if eo_stack is not None:
                torch.save(eo_stack, os.path.join(out_roi_path, "EO.pt"))
                print("  Saved EO.pt")
            else:
                print("  No EO .tif files found.")
        else:
            print("  EO folder not found.")

        # SAR
        sar_stack = read_and_process_images_torch(s1_roi_path, apply_denoise=True)
        if sar_stack is not None:
            torch.save(sar_stack, os.path.join(out_roi_path, "SAR.pt"))
            print("  Saved SAR.pt")
        else:
            print("  No SAR .tif files found.")

    print("\n✅ All ROIs processed. Preprocessed data saved in:", output_folder)

if __name__ == "__main__":
    s1_folder = r"ROIs2017_winter_s1\ROIs2017_winter"
    s2_folder = r"ROIs2017_winter_s2\ROIs2017_winter"
    output_folder = "output"

    process_all_torch(s1_folder, s2_folder, output_folder)
    print(f"\n✅ Processing complete. Outputs saved in '{output_folder}'.")

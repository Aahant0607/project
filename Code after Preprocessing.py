import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ========== Metrics ==========

def compute_ssim_psnr(pred, target):
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    channels = pred_np.shape[0]
    ssim_total, psnr_total = 0, 0
    for c in range(channels):
        ssim_total += ssim(pred_np[c], target_np[c], data_range=1.0)
        psnr_total += psnr(pred_np[c], target_np[c], data_range=1.0)
    return ssim_total / channels, psnr_total / channels

def compute_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()

def compute_ndvi(pred, red_idx=0, nir_idx=3):
    red = pred[red_idx]
    nir = pred[nir_idx]
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi

def compute_evi(pred, red_idx=0, nir_idx=3, blue_idx=2):
    red = pred[red_idx]
    nir = pred[nir_idx]
    blue = pred[blue_idx]
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    return evi

def compute_savi(pred, red_idx=0, nir_idx=3, L=0.5):
    red = pred[red_idx]
    nir = pred[nir_idx]
    savi = ((nir - red) / (nir + red + L)) * (1 + L)
    return savi

def compute_metrics(pred, target, band_order="B4,B3,B2,B8"):
    metrics = {}
    ssim_val, psnr_val = compute_ssim_psnr(pred, target)
    rmse_val = compute_rmse(pred, target)
    metrics["SSIM"] = ssim_val
    metrics["PSNR"] = psnr_val
    metrics["RMSE"] = rmse_val

    band_list = band_order.split(",")
    try:
        red_idx = band_list.index("B4")
        nir_idx = band_list.index("B8")
        blue_idx = band_list.index("B2")
        ndvi_pred = compute_ndvi(pred, red_idx, nir_idx)
        ndvi_target = compute_ndvi(target, red_idx, nir_idx)
        metrics["NDVI_MAE"] = torch.mean(torch.abs(ndvi_pred - ndvi_target)).item()

        evi_pred = compute_evi(pred, red_idx, nir_idx, blue_idx)
        evi_target = compute_evi(target, red_idx, nir_idx, blue_idx)
        metrics["EVI_MAE"] = torch.mean(torch.abs(evi_pred - evi_target)).item()

        savi_pred = compute_savi(pred, red_idx, nir_idx)
        savi_target = compute_savi(target, red_idx, nir_idx)
        metrics["SAVI_MAE"] = torch.mean(torch.abs(savi_pred - savi_target)).item()

    except ValueError:
        print("⚠️ NDVI/EVI/SAVI skipped — required bands not found in band_order.")
    return metrics

# ========== Dataset ==========

class SARToEODataset(Dataset):
    """
    Loads paired SAR and EO .pt tensors from folders.
    Selects EO bands per band_order (comma-separated bands like B4,B3,B2,B8).
    Assumes SAR has fixed shape [1,H,W], EO full bands [C,H,W].
    """
    band_to_index = {
        "B2": 0, "B3": 1, "B4": 2, "B5": 3, "B8": 4, "B11": 5  # Adjust indices based on your EO data structure
    }

    def __init__(self, root_dir, band_order="B4,B3,B2", transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.samples = sorted(os.listdir(root_dir))
        self.band_order = band_order.split(",")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder = os.path.join(self.root_dir, self.samples[idx])
        sar_path = os.path.join(folder, "SAR.pt")
        eo_path = os.path.join(folder, "EO.pt")
        sar = torch.load(sar_path)  # [1,H,W]
        eo_full = torch.load(eo_path)  # [C,H,W]

        # Select EO bands as per band_order
        eo_bands = []
        for b in self.band_order:
            if b in self.band_to_index and self.band_to_index[b] < eo_full.shape[0]:
                eo_bands.append(eo_full[self.band_to_index[b]])
            else:
                raise ValueError(f"Band {b} not found in EO data")

        eo = torch.stack(eo_bands, dim=0)  # [len(band_order),H,W]

        if self.transform:
            sar = self.transform(sar)
            eo = self.transform(eo)

        return sar.float(), eo.float()

# ========== UNet for SAR denoising ==========

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        self.enc1 = nn.Sequential(CBR(in_channels, 64), CBR(64, 64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(CBR(256, 512), CBR(512, 512))
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(CBR(512, 256), CBR(256, 256))
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        up3 = self.up3(b)
        d3 = self.dec3(torch.cat([up3, e3], dim=1))
        up2 = self.up2(d3)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))
        up1 = self.up1(d2)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))
        out = self.outc(d1)
        return out

# ========== CycleGAN Generator and Discriminator (ResNet + PatchGAN) ==========

# Resnet Generator
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        assert(n_blocks >= 0)
        super().__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        # Resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.conv_block(x)

# PatchGAN Discriminator
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

# ========== Training Pipeline ==========

def train_pipeline(
    root_dir="/kaggle/input/ak-cycle-dataset/output",
    epochs_unet=5,
    epochs_cyclegan=5,
    batch_size=4,
    lr_unet=1e-3,
    lr_gan=2e-4,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"Using device: {device}")

    # Band orders to test with CycleGAN models
    band_orders = ["B4,B3,B2", "B8,B4,B3,B2", "B4,B3,B2,B8"]
    num_models = len(band_orders)

    # Load Datasets and Dataloaders for each band_order
    dataloaders = {}
    datasets = {}
    for bo in band_orders:
        ds = SARToEODataset(root_dir=root_dir, band_order=bo)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        datasets[bo] = ds
        dataloaders[bo] = dl

    # ---------- Train UNet (SAR denoising) ----------

    print("\n==> Training UNet for SAR denoising...")
    unet = UNet(in_channels=1, out_channels=1).to(device)
    optimizer_unet = optim.Adam(unet.parameters(), lr=lr_unet)
    criterion_l1 = nn.L1Loss()

    unet_losses = []
    for epoch in range(epochs_unet):
        unet.train()
        epoch_loss = 0
        for sar, _ in dataloaders["B4,B3,B2"]:  # SAR input only, EO ignored here
            sar = sar.to(device)
            optimizer_unet.zero_grad()
            denoised = unet(sar)
            loss = criterion_l1(denoised, sar)
            loss.backward()
            optimizer_unet.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataloaders["B4,B3,B2"])
        unet_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs_unet} UNet Loss: {epoch_loss:.4f}")

    # ---------- Train CycleGANs ----------

    print("\n==> Training CycleGAN models...")
    cyclegan_models = {}
    cyclegan_losses = {}

    for bo in band_orders:
        print(f"\n-- CycleGAN for band_order: {bo} --")

        dataset = datasets[bo]
        dataloader = dataloaders[bo]

        input_nc = 1       # SAR channels
        output_nc = len(bo.split(","))  # EO bands count

        G = ResnetGenerator(input_nc, output_nc).to(device)
        F = ResnetGenerator(output_nc, input_nc).to(device)
        D_Y = NLayerDiscriminator(output_nc).to(device)
        D_X = NLayerDiscriminator(input_nc).to(device)

        opt_G = optim.Adam(list(G.parameters()) + list(F.parameters()), lr=lr_gan, betas=(0.5, 0.999))
        opt_D_Y = optim.Adam(D_Y.parameters(), lr=lr_gan, betas=(0.5, 0.999))
        opt_D_X = optim.Adam(D_X.parameters(), lr=lr_gan, betas=(0.5, 0.999))

        criterion_GAN = nn.MSELoss()
        criterion_cycle = nn.L1Loss()
        criterion_identity = nn.L1Loss()

        lambda_cycle = 10.0
        lambda_id = 0.5

        losses = {
            "G_GAN": [], "F_GAN": [], "cycle": [], "identity": []
        }

        for epoch in range(epochs_cyclegan):
            G.train(); F.train(); D_Y.train(); D_X.train()
            total_G_GAN = total_F_GAN = total_cycle = total_id = 0
            for sar, eo in dataloader:
                sar = sar.to(device)
                eo = eo.to(device)

                # Adversarial ground truths
                valid = torch.ones(sar.size(0), 1, 30, 30, device=device)  # PatchGAN output size ~30x30 for 224x224 input
                fake = torch.zeros_like(valid)

                ###### Train Generators G and F ######

                opt_G.zero_grad()

                # Identity loss
                idt_y = G(eo)
                loss_idt_y = criterion_identity(idt_y, eo) * lambda_cycle * lambda_id
                idt_x = F(sar)
                loss_idt_x = criterion_identity(idt_x, sar) * lambda_cycle * lambda_id

                # GAN loss G: G(SAR) -> EO
                fake_eo = G(sar)
                pred_fake = D_Y(fake_eo)
                loss_GAN_G = criterion_GAN(pred_fake, valid)

                # GAN loss F: F(EO) -> SAR
                fake_sar = F(eo)
                pred_fake_F = D_X(fake_sar)
                loss_GAN_F = criterion_GAN(pred_fake_F, valid)

                # Cycle loss
                rec_sar = F(fake_eo)
                loss_cycle_sar = criterion_cycle(rec_sar, sar)
                rec_eo = G(fake_sar)
                loss_cycle_eo = criterion_cycle(rec_eo, eo)
                loss_cycle_total = loss_cycle_sar + loss_cycle_eo

                # Total loss
                loss_G = loss_GAN_G + loss_GAN_F + lambda_cycle * loss_cycle_total + loss_idt_y + loss_idt_x
                loss_G.backward()
                opt_G.step()

                ###### Train Discriminator D_Y ######

                opt_D_Y.zero_grad()
                pred_real = D_Y(eo)
                loss_D_real = criterion_GAN(pred_real, valid)
                pred_fake_detached = D_Y(fake_eo.detach())
                loss_D_fake = criterion_GAN(pred_fake_detached, fake)
                loss_D_Y = (loss_D_real + loss_D_fake) * 0.5
                loss_D_Y.backward()
                opt_D_Y.step()

                ###### Train Discriminator D_X ######

                opt_D_X.zero_grad()
                pred_real_x = D_X(sar)
                loss_D_real_x = criterion_GAN(pred_real_x, valid)
                pred_fake_x = D_X(fake_sar.detach())
                loss_D_fake_x = criterion_GAN(pred_fake_x, fake)
                loss_D_X = (loss_D_real_x + loss_D_fake_x) * 0.5
                loss_D_X.backward()
                opt_D_X.step()

                total_G_GAN += loss_GAN_G.item()
                total_F_GAN += loss_GAN_F.item()
                total_cycle += loss_cycle_total.item()
                total_id += (loss_idt_y.item() + loss_idt_x.item())

            avg_G_GAN = total_G_GAN / len(dataloader)
            avg_F_GAN = total_F_GAN / len(dataloader)
            avg_cycle = total_cycle / len(dataloader)
            avg_id = total_id / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs_cyclegan}: G_GAN={avg_G_GAN:.4f}, F_GAN={avg_F_GAN:.4f}, cycle={avg_cycle:.4f}, id={avg_id:.4f}")

            losses["G_GAN"].append(avg_G_GAN)
            losses["F_GAN"].append(avg_F_GAN)
            losses["cycle"].append(avg_cycle)
            losses["identity"].append(avg_id)

        cyclegan_models[bo] = (G, F, D_Y, D_X)
        cyclegan_losses[bo] = losses

    # ---------- Evaluation and Visualization ----------

    print("\n==> Evaluating and visualizing results for 5 samples...")

    def visualize_samples(sar, eo_gt, eo_pred, band_order):
        # sar: [1,H,W], eo_gt and eo_pred: [C,H,W]
        import matplotlib.pyplot as plt

        n_bands = eo_gt.shape[0]
        fig, axes = plt.subplots(3, n_bands + 1, figsize=(3*(n_bands+1), 9))
        fig.suptitle(f"Visualization Band Order: {band_order}")

        for b in range(n_bands):
            axes[0, b].imshow(eo_gt[b].cpu(), cmap='gray')
            axes[0, b].set_title(f"GT {band_order.split(',')[b]}")
            axes[0, b].axis('off')

            axes[1, b].imshow(eo_pred[b].cpu().detach(), cmap='gray')
            axes[1, b].set_title(f"Pred {band_order.split(',')[b]}")
            axes[1, b].axis('off')

            diff = torch.abs(eo_gt[b] - eo_pred[b])
            axes[2, b].imshow(diff.cpu(), cmap='hot')
            axes[2, b].set_title("Abs Diff")
            axes[2, b].axis('off')

        # SAR image
        axes[0, -1].imshow(sar[0].cpu(), cmap='gray')
        axes[0, -1].set_title("SAR input")
        axes[0, -1].axis('off')
        axes[1, -1].axis('off')
        axes[2, -1].axis('off')

        plt.tight_layout()
        plt.show()

    # Visualize UNet denoising on SAR for 5 samples (no EO)
    print("\nUNet SAR denoising visualization:")
    unet.eval()
    dataset_unet = datasets["B4,B3,B2"]  # dummy for data, EO ignored
    for i in range(min(5, len(dataset_unet))):
        sar, _ = dataset_unet[i]
        with torch.no_grad():
            sar_denoised = unet(sar.unsqueeze(0).to(device)).squeeze(0).cpu()
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.title("Original SAR")
        plt.imshow(sar[0], cmap='gray')
        plt.axis("off")
        plt.subplot(1,2,2)
        plt.title("Denoised SAR (UNet)")
        plt.imshow(sar_denoised[0], cmap='gray')
        plt.axis("off")
        plt.show()

    # Visualize CycleGAN outputs + print metrics
    for bo in band_orders:
        print(f"\nVisualizing and computing metrics for CycleGAN band_order: {bo}")
        G, _, _, _ = cyclegan_models[bo]
        dataset = datasets[bo]

        all_metrics = []
        G.eval()
        for i in range(min(5, len(dataset))):
            sar, eo_gt = dataset[i]
            sar = sar.unsqueeze(0).to(device)
            with torch.no_grad():
                eo_pred = G(sar).squeeze(0).cpu().clamp(0, 1)

            metrics = compute_metrics(eo_pred, eo_gt, band_order=bo)
            all_metrics.append(metrics)

            print(f"Sample {i+1} metrics:", {k: round(v,4) for k,v in metrics.items()})
            visualize_samples(sar.squeeze(0).cpu(), eo_gt, eo_pred, bo)

        # Print average metrics for this model
        avg_metrics = {}
        for k in all_metrics[0].keys():
            avg_metrics[k] = np.mean([m[k] for m in all_metrics])
        print(f"Average metrics for band_order {bo}:", {k: round(v,4) for k,v in avg_metrics.items()})

    # ---------- Plot losses ----------

    plt.figure(figsize=(12,5))
    plt.plot(unet_losses, label="UNet L1 Loss")
    plt.title("UNet Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    for bo in band_orders:
        losses = cyclegan_losses[bo]
        plt.figure(figsize=(12,5))
        plt.plot(losses["G_GAN"], label="G GAN Loss")
        plt.plot(losses["F_GAN"], label="F GAN Loss")
        plt.plot(losses["cycle"], label="Cycle Loss")
        plt.plot(losses["identity"], label="Identity Loss")
        plt.title(f"CycleGAN Training Losses for band_order {bo}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    train_pipeline()

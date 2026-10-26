#!/usr/bin/env python3
"""DRIPS PVS inference using SIAM for native-space ICV / CSF masking only."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.networks.nets import AttentionUnet
from nibabel.processing import resample_from_to

ROOT = Path("/opt/drips")
# SIAM v3 labels used by the original DRIPS release: CSF and ventricular CSF.
SIAM_CSF = (3, 4)


class Laplacian(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor, spacing: tuple[float, float, float]) -> torch.Tensor:
        sp = torch.as_tensor(spacing, device=x.device, dtype=x.dtype)
        out = []
        for batch_index in range(x.shape[0]):
            if self.dims == 3:
                wz, wy, wx = 1.0 / (sp * sp + 1e-6)
                kernel = torch.zeros((1, 1, 3, 3, 3), device=x.device, dtype=x.dtype)
                kernel[0, 0, 0, 1, 1] = kernel[0, 0, 2, 1, 1] = wz
                kernel[0, 0, 1, 0, 1] = kernel[0, 0, 1, 2, 1] = wy
                kernel[0, 0, 1, 1, 0] = kernel[0, 0, 1, 1, 2] = wx
                kernel[0, 0, 1, 1, 1] = -2.0 * (wz + wy + wx)
                scale = 2.0 * (wz + wy + wx)
                value = F.conv3d(
                    F.pad(x[batch_index : batch_index + 1].float(), (1, 1, 1, 1, 1, 1), mode="replicate"),
                    kernel,
                )
            else:
                wz, wx = 1.0 / (sp[0] * sp[0] + 1e-6), 1.0 / (sp[2] * sp[2] + 1e-6)
                kernel = torch.zeros((1, 1, 3, 3), device=x.device, dtype=x.dtype)
                kernel[0, 0, 0, 1] = kernel[0, 0, 2, 1] = wz
                kernel[0, 0, 1, 0] = kernel[0, 0, 1, 2] = wx
                kernel[0, 0, 1, 1] = -2.0 * (wz + wx)
                scale = 2.0 * (wz + wx)
                value = F.conv2d(
                    F.pad(x[batch_index : batch_index + 1].float(), (1, 1, 1, 1), mode="replicate"),
                    kernel,
                )
            out.append(F.relu(-value / (scale + 1e-6)))
        return torch.cat(out).to(x.dtype)


class DRIPSNet(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        convolution = nn.Conv3d if dims == 3 else nn.Conv2d
        self.laplacian = Laplacian(dims)
        self.unet = AttentionUnet(
            spatial_dims=dims,
            in_channels=2,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            kernel_size=3,
            up_kernel_size=3,
        )
        self.refine = nn.Sequential(
            convolution(3, 32, 3, padding=1), nn.LeakyReLU(0.1, inplace=True), convolution(32, 1, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, spacing: tuple[float, float, float]) -> torch.Tensor:
        laplacian = self.laplacian(x, spacing)
        features = torch.cat((x, laplacian), 1)
        return self.refine(torch.cat((features, self.unet(features)), 1))


def run(command: list[object]) -> None:
    subprocess.run([str(value) for value in command], check=True)


def load_image(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI image, got {data.ndim}D: {path}")
    return image, data


def resample_to_native(path: Path, reference: nib.Nifti1Image) -> np.ndarray:
    image = nib.load(str(path))
    if image.shape == reference.shape and np.allclose(image.affine, reference.affine):
        return np.asarray(image.dataobj)
    return resample_from_to(image, reference, order=0).get_fdata()


def run_siam(image: Path, work_dir: Path, threads: int) -> Path:
    """Generate native-space whole-head labels with the packaged SIAM install."""
    siam_dir = work_dir / "siam"
    siam_dir.mkdir(parents=True, exist_ok=True)
    # In current SIAM, model 3 is the default and is requested as -m 0.
    # Do not force a device: SIAM selects CUDA when it is available.
    run(["siam-pred", "-i", image, "-o", siam_dir, "-m", "0", "-nbthread", threads])
    outputs = list(siam_dir.rglob("*.nii")) + list(siam_dir.rglob("*.nii.gz"))
    if not outputs:
        raise RuntimeError("SIAM completed but did not produce a NIfTI segmentation.")
    return max(outputs, key=lambda candidate: candidate.stat().st_mtime)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return torch.device("cpu")
    capability = torch.cuda.get_device_capability()
    architecture = f"sm_{capability[0]}{capability[1]}"
    supported = set(torch.cuda.get_arch_list())
    if architecture not in supported:
        message = f"Installed PyTorch does not support {architecture}; supported architectures: {sorted(supported)}"
        if requested == "cuda":
            raise RuntimeError(message)
        warnings.warn(message + "; falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda")


def load_model(path: Path, dims: int, device: torch.device) -> DRIPSNet:
    model = DRIPSNet(dims).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def predict(
    image: nib.Nifti1Image,
    data: np.ndarray,
    icv: np.ndarray,
    csf: np.ndarray,
    model: DRIPSNet,
    dims: int,
    contrast: str,
    threshold: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    mask = icv.astype(bool)
    valid = np.isfinite(data) & mask
    if valid.sum() == 0:
        raise ValueError("The ICV mask has no finite voxels in the input image.")
    low, high = np.quantile(data[valid], (0.05, 0.95))
    if high <= low:
        raise ValueError("Input image has no usable intensity range inside the ICV mask.")
    normalized = np.clip((data - low) / (high - low), 0.0, 1.0)
    if contrast == "hypointense":
        normalized = 1.0 - normalized
    normalized[~mask] = 0.0

    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    tensor = torch.from_numpy(normalized[None, None]).to(device)
    with torch.inference_mode():
        if dims == 3:
            predictions = []
            for flips in ((), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)):
                augmented = torch.flip(tensor, flips) if flips else tensor
                logits = sliding_window_inference(
                    augmented, (128, 128, 128), 1, lambda batch: model(batch, spacing), overlap=0.25, mode="gaussian"
                )
                probability = torch.sigmoid(logits)
                predictions.append(torch.flip(probability, flips) if flips else probability)
            probability = torch.stack(predictions).mean(0)
        else:
            slices = tensor[0].permute(2, 0, 1, 3).contiguous()
            batches = []
            for start in range(0, len(slices), 16):
                batch = slices[start : start + 16]
                predictions = []
                for flips in ((), (2,), (3,), (2, 3)):
                    augmented = torch.flip(batch, flips) if flips else batch
                    logits = sliding_window_inference(
                        augmented, (128, 128), 1, lambda value: model(value, spacing), overlap=0.25, mode="gaussian"
                    )
                    probability = torch.sigmoid(logits)
                    predictions.append(torch.flip(probability, flips) if flips else probability)
                batches.append(torch.stack(predictions).mean(0))
            probability = torch.cat(batches).permute(1, 2, 0, 3).unsqueeze(0)

    output = probability[0, 0].cpu().numpy()
    output[~mask] = 0.0
    output[csf.astype(bool)] = 0.0
    return output, (output >= threshold).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="DRIPS PVS inference with SIAM-derived ICV masking.")
    parser.add_argument("--in", dest="image", required=True, help="Input 3D NIfTI image.")
    parser.add_argument("--out", required=True, help="Output binary PVS NIfTI path.")
    parser.add_argument("--posterior", required=True, help="Output PVS probability NIfTI path.")
    parser.add_argument("--icv-mask", help="Optional native-space ICV mask. If omitted, SIAM creates it.")
    parser.add_argument("--save-icv", help="Optional path to save the ICV mask used for inference.")
    parser.add_argument("--model", choices=("auto", "2d", "3d"), default="auto")
    parser.add_argument("--pvs-contrast", choices=("hyperintense", "hypointense"), default="hyperintense")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--work-dir", help="Optional directory for retained SIAM outputs.")
    arguments = parser.parse_args()

    image_path = Path(arguments.image).resolve()
    image, data = load_image(image_path)
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    mode = arguments.model if arguments.model != "auto" else ("3d" if spacing.max() / spacing.min() <= 1.5 else "2d")
    dims = 3 if mode == "3d" else 2

    owns_work_dir = arguments.work_dir is None
    work_dir = Path(tempfile.mkdtemp(prefix="drips_siam_")) if owns_work_dir else Path(arguments.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        if arguments.icv_mask:
            icv = resample_to_native(Path(arguments.icv_mask), image) > 0
            csf = np.zeros(image.shape, dtype=bool)
        else:
            siam_segmentation = run_siam(image_path, work_dir, arguments.threads)
            siam_native = resample_to_native(siam_segmentation, image)
            icv = siam_native > 0
            csf = np.isin(siam_native, SIAM_CSF)
        if arguments.save_icv:
            nib.save(nib.Nifti1Image(icv.astype(np.uint8), image.affine, image.header), arguments.save_icv)

        device = choose_device(arguments.device)
        model_path = ROOT / "models" / ("unet_pvs_3d.pt" if dims == 3 else "unet_pvs_2d.pt")
        model = load_model(model_path, dims, device)
        posterior, segmentation = predict(
            image, data, icv, csf, model, dims, arguments.pvs_contrast, arguments.threshold, device
        )
        nib.save(nib.Nifti1Image(posterior.astype(np.float32), image.affine, image.header), arguments.posterior)
        nib.save(nib.Nifti1Image(segmentation, image.affine, image.header), arguments.out)
    finally:
        if owns_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

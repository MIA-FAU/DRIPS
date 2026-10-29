# DRIPS

**Domain Randomisation for Image-based Perivascular Spaces Segmentation**

DRIPS is a deep-learning framework for automated segmentation of perivascular spaces (PVS) on structural MRI. It was trained with domain randomisation on synthetic data and supports T1w, T2w, FLAIR, isotropic, and anisotropic scans. It produces a binary PVS segmentation and a float32 posterior-probability map in the input image space.

DRIPS is distributed as an Apptainer/SIF image. Docker is not supported.

## Requirements

- Apptainer or Singularity
- An NVIDIA GPU and a compatible NVIDIA driver are recommended. CPU inference is supported but can be substantially slower.
- Input image in NIfTI format (`.nii` or `.nii.gz`)

## Download

Download the current `drips_blackwell.sif` file from Google Drive: https://drive.google.com/file/d/1Oa08T3h6fMeLlc5omLL00sj8UU2mlnYc/view?usp=sharing .

## Quick start

Place your input image in a local directory, for example:

```text
cases/
└── t1w.nii.gz
```

Run the container with GPU support:

```bash
apptainer run --nv \
  --bind "$HOME/cases:/data" \
  drips_blackwell.sif \
  --in /data/t1w.nii.gz \
  --out /data/pvs.nii.gz \
  --posterior /data/pvs_probability.nii.gz \
  --model auto \
  --pvs-contrast hypointense
```

This command creates:

```text
cases/
├── pvs.nii.gz
└── pvs_probability.nii.gz
```

`pvs.nii.gz` is the binary PVS segmentation and `pvs_probability.nii.gz` is the posterior probability map.

## ICV mask and SIAM

If no ICV mask is supplied, DRIPS invokes the included `siam-pred` command to generate a native-space ICV mask and exclude SIAM-labelled CSF/ventricular CSF regions. **We highly recommend this option.**

To retain the generated mask:

```bash
apptainer run --nv \
  --bind "$HOME/cases:/data" \
  drips_blackwell.sif \
  --in /data/t1w.nii.gz \
  --out /data/pvs.nii.gz \
  --posterior /data/pvs_probability.nii.gz \
  --save-icv /data/icv_siam.nii.gz \
  --model auto \
  --pvs-contrast hypointense
```

Alternatively, provide an existing native-space ICV mask:

```bash
apptainer run --nv \
  --bind "$HOME/cases:/data" \
  drips_blackwell.sif \
  --in /data/t1w.nii.gz \
  --icv-mask /data/icv_mask.nii.gz \
  --out /data/pvs.nii.gz \
  --posterior /data/pvs_probability.nii.gz \
  --model auto \
  --pvs-contrast hypointense
```

## Model selection and 2D orientation

`--model auto` selects the 3D model when:

```text
max(voxel spacing) / min(voxel spacing) ≤ 1.5
```

and otherwise selects the 2D model. Thus, a `1 × 1 × 2 mm` scan is processed with the 2D model. Use `--model 3d` or `--model 2d` to override this rule.

The 2D model processes slices orthogonal to one NIfTI array axis. The default, `--2d-plane 2`, treats the third array axis as the superior--inferior axis, which is normally appropriate for axial acquisitions. Use `--2d-plane 0` or `--2d-plane 1` only when the corresponding input array axis is superior--inferior. The result is transformed back to the original image orientation before it is written.

## Options

| Option | Description |
| --- | --- |
| `--in PATH` | Required input 3D NIfTI image. |
| `--out PATH` | Required output binary PVS segmentation. |
| `--posterior PATH` | Required output PVS posterior-probability map. |
| `--icv-mask PATH` | Optional native-space ICV mask. Skips SIAM processing. |
| `--save-icv PATH` | Save the ICV mask produced by SIAM. |
| `--model auto\|2d\|3d` | Model choice; default: `auto`. |
| `--2d-plane 0\|1\|2` | Superior--inferior array-axis index for 2D inference; default: `2`. |
| `--pvs-contrast hyperintense\|hypointense` | Required. Use `hyperintense` for bright PVS (typically T2w) and `hypointense` for dark PVS (typically T1w). Hypointense inputs are inverted before inference. |
| `--threshold FLOAT` | Final binary threshold applied to the posterior; default: `0.5`. |
| `--device auto\|cuda\|cpu` | Compute device; default: `auto`. |
| `--threads INT` | SIAM preprocessing threads; default: `4`. |

## GPU check

Before first inference on a new system, verify that the container can access the GPU:

```bash
apptainer exec --nv drips_blackwell.sif python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_arch_list())
assert torch.cuda.is_available()
print(torch.randn(8, 8, device="cuda").sum())
PY
```

## Citation

If you use DRIPS, please cite:

> Bitar L, Díaz M, Duarte Coello R, et al., Bernal J.  
> *DRIPS: Domain Randomisation for Image-based Perivascular Spaces Segmentation.*  
> medRxiv. 2025. https://doi.org/10.1101/2025.10.22.25337423

## Support

Please report issues through the [GitHub issue tracker](https://github.com/MIA-FAU/DRIPS/issues). Include the command used, relevant error output, and the Apptainer version. Do not upload participant images or other sensitive data.

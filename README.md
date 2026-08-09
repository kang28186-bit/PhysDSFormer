# PhysDSFormer

PyTorch implementation of **PhysDSFormer: A Lightweight Physics-Aware Transformer for Real-Time Underwater Image Enhancement**.

The repository provides model training, evaluation, and inference code. Dataset images and generated experiment artifacts are kept outside the source tree.

## Method

PhysDSFormer combines three components:

- **Physics-aware Patch Embedding (PhysPE):** an OP Reconstruction module estimates channel-wise gain and bias maps from a simplified underwater imaging model, performs preliminary restoration, and maps the restored RGB image into shallow features with a 5×5 convolution.
- **Depthwise Separable Convolution (DSC):** depthwise spatial filtering followed by 1×1 pointwise fusion complements window self-attention with efficient local texture modeling.
- **Multi-color-space structural loss:** RGB, LAB, LCH, and SSIM supervision jointly constrain pixel fidelity, luminance, chroma, hue, and local structure.

The tiny configuration contains 51,554 trainable parameters (approximately 0.052 M).

### Reproducible implementation settings

The published preset is `physdsformer-tiny`: stage dimensions `[24, 24, 24, 24]`, depths `[2, 2, 2, 2]`, heads `[2, 2, 2, 1]`, MLP ratios `[2, 2, 2, 2]`, attention ratios `[0, 1/2, 1, 0]`, and window size 8. OP Reconstruction uses seven hidden channels. Each DSC uses a 3×3 reflected depthwise convolution, a 1×1 pointwise convolution, and ReLU before the attention/FFN path. The decoder head predicts one shared residual gain and three RGB bias channels.

The training objective is `0.001 L_RGB + 100 L_LAB + 0.1 L_LCH + (1 - SSIM)`. Color-statistic terms use a per-image, per-channel 32-bin triangular soft histogram. LAB channels are normalized as `L/100`, `(A+128)/255`, and `(B+128)/255`; chroma is divided by `sqrt(2)×128`; hue uses a low-chroma-weighted circular cosine distance. SSIM uses an 11×11 Gaussian window with sigma 1.5. These operations run in FP32 even when mixed precision is enabled for the network.

## Repository layout

```text
PhysDSFormer/
├── configs/                 # Dataset-specific training settings
├── datasets/                # Paired image interface only; no images
├── losses/                  # RGB/LAB/LCH/SSIM objective
├── models/                  # PhysDSFormer model definitions
├── tools/smoke_test.py      # Forward, loss, and backward sanity check
├── train.py                 # Training entry point
├── evaluate.py              # Paired-dataset evaluation
├── infer.py                 # Single-image or directory inference
├── environment.yml
└── requirements.txt
```

## Environment

The manuscript experiments used Python 3.7, PyTorch 1.10.2, CUDA 11.3, and cuDNN 8.2. A matching Conda environment can be created with:

```bash
conda env create -f environment.yml
conda activate physdsformer
```

For an existing PyTorch environment:

```bash
pip install -r requirements.txt
```

## Sanity check

Run the self-contained CPU smoke test:

```bash
python tools/smoke_test.py
```

It instantiates the tiny model, performs a forward pass, evaluates the complete loss, and verifies backward propagation. No dataset or checkpoint is needed.

## Inference

Place a checkpoint at a local path and run:

```bash
python infer.py \
  --model physdsformer-tiny \
  --checkpoint checkpoints/uieb/physdsformer-tiny.pth \
  --input path/to/input_images \
  --output results/uieb
```

The checkpoint may be a plain state dictionary or a dictionary containing `state_dict`. Checkpoints must match the architecture in this release. Input dimensions are padded internally to safe multiples and the output is cropped back to the original size.

## Training and evaluation

Train on one of the supported paired datasets:

```bash
python train.py \
  --model physdsformer-tiny \
  --dataset UIEB \
  --data-root data \
  --output checkpoints
```

Training selects checkpoints on `data/<dataset>/val`; the held-out `test` split is used only by `evaluate.py`. Use `--validation-split <name>` if your validation directory has a different name.

Resume an interrupted run with `--resume checkpoints/uieb/physdsformer-tiny-last.pth`; model, optimizer, scheduler, scaler, epoch, and best validation score are restored. The file without the `-last` suffix remains the best validation checkpoint.

Evaluate a trained checkpoint and optionally save enhanced images:

```bash
python evaluate.py \
  --model physdsformer-tiny \
  --dataset UIEB \
  --data-root data \
  --checkpoint checkpoints/uieb/physdsformer-tiny.pth \
  --save-images
```

## Dataset preparation

Dataset images are intentionally excluded. Download each dataset from its official source:

- [UIEB](https://li-chongyi.github.io/proj_benchmark.html)
- [LSUI](https://lintaopeng.github.io/code/)
- [EUVP](https://irvlab.cs.umn.edu/resources/euvp-dataset)

Arrange paired images as follows, using identical file names in `cond` and `gt`:

```text
data/
└── UIEB/
    ├── train/
    │   ├── cond/
    │   └── gt/
    ├── val/
    │   ├── cond/
    │   └── gt/
    └── test/
        ├── cond/
        └── gt/
```

The same structure applies to LSUI and EUVP. The loader resizes images to 256×256, normalizes values to [−1, 1], and supports paired rotation and horizontal-flip augmentation.

## Experimental scope

The provided JSON files record the model-selection split, complete loss weights, and the principal training settings: AdamW, batch size 8, initial learning rate 2×10⁻⁴, cosine decay to zero, 300 epochs, and 256×256 inputs. Use `--seed` to repeat an experiment under multiple seeds. Reported metrics depend on the data split, random seeds, hardware/software versions, and selected checkpoint; this repository does not hard-code benchmark scores.

## Citation

```bibtex
@article{dai2026physdsformer,
  title   = {PhysDSFormer: A Lightweight Physics-Aware Transformer for Real-Time Underwater Image Enhancement},
  author  = {Dai, Yuntao and Shi, Dada and Liu, Liqiang},
  year    = {2026},
  note    = {Manuscript submitted to The Visual Computer}
}
```

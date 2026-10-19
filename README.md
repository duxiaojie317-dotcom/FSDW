# FSDW

Frequency-Spatial Dual-Domain Imperceptible Watermarking for Robust Image Datasets Copyright Protection

Pretrained-model evaluation for MNIST, CIFAR-10 and CIFAR-100. This release contains no training code or datasets. Only the CIFAR-10 checkpoint is provided; evaluating the other datasets requires their matching checkpoints and configurations.

## Setup

Use Python 3.11:

```bash
pip install -r requirements.txt
```

For GPU evaluation, install a compatible PyTorch/torchvision CUDA build. CPU evaluation is also supported.

## Model and data

Download `FSDW_cifar10_weights.zip` from [Releases](https://github.com/duxiaojie317-dotcom/FSDW/releases) and extract it into `checkpoints/`:

```text
checkpoints/cifar10/
├── classifier_model.pth
├── freq_mask.pth
├── config.json
└── SHA256SUMS.txt
```

Provide the CIFAR-10 Python-format dataset in `/path/to/data/cifar-10-batches-py/`. Replace `/path/to/data` in the command below with the actual directory. Add `--download` only if you want torchvision to download the dataset.

## Evaluate

Run from the repository root:

```bash
python test.py --config configs/cifar10.json --data-root /path/to/data --num-workers 0 --model-dir checkpoints/cifar10 --results results/cifar10_metrics.json
```

Both the classifier and its matching frequency mask are required. Keep the dataset, embedding strength and target class consistent with the checkpoint. Results are printed and saved as JSON. Existing result files are not overwritten; use a new filename for another run.

The supplied CIFAR-10 checkpoint was evaluated on all 10,000 test images:

| CA | WSR | JPEG | Blur | Noise | Avg |
|---:|---:|---:|---:|---:|---:|
| 94.73 | 99.78 | 100.00 | 100.00 | 99.96 | 99.99 |

These are results for the supplied checkpoint, not an exact copy of the paper's Table 2 values. Results may vary slightly across environments.

CA is clean accuracy. WSR is the fraction of watermarked test images classified as the target class, including images originally in that class. JPEG, Blur and Noise report WSR after the corresponding attacks; Avg is their arithmetic mean. All values are percentages. Defaults: JPEG quality 50, Gaussian blur kernel 3 (sigma 0.8), Gaussian noise standard deviation 0.05, evaluation seed 12345.

For MNIST or CIFAR-100, select the corresponding JSON file in `configs/` and provide that dataset's own classifier, frequency mask and matching settings. CIFAR-10 weights cannot be used for these datasets.

## Files

- `test.py`: full-test-set evaluation.
- `core/`: model architecture, evaluation-time watermark generation, attacks and utilities.
- `configs/`: evaluation settings.

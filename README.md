# FSDW

Frequency-Spatial Dual-Domain Imperceptible Watermarking for Robust Image Datasets Copyright Protection

Training and evaluation code for our watermarking method on MNIST, CIFAR-10 and CIFAR-100.
Datasets, trained weights and experiment outputs are not included.

## Setup

Use Python 3.11 and install the dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, use a PyTorch/torchvision build compatible with your CUDA environment.

## Training

Run commands from the repository root. Set `--data-root` to your dataset directory, containing `MNIST/raw/`, `cifar-10-batches-py/` or `cifar-100-python/` as appropriate.

```bash
python train.py --config configs/mnist.json --data-root /path/to/data --num-workers 0 --output-dir outputs/mnist
python train.py --config configs/cifar10.json --data-root /path/to/data --num-workers 0 --output-dir outputs/cifar10
python train.py --config configs/cifar100.json --data-root /path/to/data --num-workers 0 --output-dir outputs/cifar100
```

Replace `/path/to/data` with your actual path. Add `--download` if the dataset needs to be downloaded and network access is available.
Training settings are in `configs/`. Use a new output directory for each run; nonempty directories are not overwritten.

## Evaluation

After training, evaluate the saved model and its matching frequency mask:

```bash
python test.py --config configs/mnist.json --data-root /path/to/data --num-workers 0 --model-dir outputs/mnist --results outputs/mnist/metrics.json
python test.py --config configs/cifar10.json --data-root /path/to/data --num-workers 0 --model-dir outputs/cifar10 --results outputs/cifar10/metrics.json
python test.py --config configs/cifar100.json --data-root /path/to/data --num-workers 0 --model-dir outputs/cifar100 --results outputs/cifar100/metrics.json
```

Use the same configuration as training. The model directory must contain `classifier_model.pth` and `freq_mask.pth`.
If you changed the training output directory, update the evaluation paths accordingly. Existing result files are not overwritten.

Evaluation reports **CA**, **WSR**, and WSR after **JPEG**, **Blur** and **Noise** attacks on the full test split.
**Avg** is the arithmetic mean of the three attacked WSR values. All metrics are percentages; WSR includes all test images, including target-class images.
Compare the resulting metrics with the Ours row in Table 2. Results may vary with the environment and random seed.

## Code

- `train.py`: training entry point.
- `test.py`: evaluation entry point.
- `core/`: watermark embedding, frequency analysis, sample selection, model and attacks.
- `configs/`: dataset-specific settings.

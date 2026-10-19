import argparse
import json
import math
from pathlib import Path


def parse_settings():
    parser = argparse.ArgumentParser(description='Evaluate FSDW pretrained models')
    parser.add_argument('--config', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--results', default=None)
    cli = parser.parse_args()
    config = json.loads(Path(cli.config).read_text(encoding='utf-8'))
    required = {'dataset', 'arch', 'eval_seed', 'eval_batch_size',
                'embed_strength', 'target_class', 'num_workers', 'data_root',
                'jpeg_quality', 'blur_kernel', 'noise_std'}
    missing = required - config.keys()
    if missing:
        parser.error(f'Missing configuration fields: {sorted(missing)}')
    config = {key: config[key] for key in required}
    config.update({key: value for key, value in vars(cli).items() if value is not None})
    config.setdefault('results', None)
    args = argparse.Namespace(**config)
    if args.dataset not in ('mnist', 'cifar10', 'cifar100'):
        parser.error('Unsupported dataset')
    if args.arch != 'resnet18':
        parser.error('Only ResNet-18 is supported')
    integer_fields = ('eval_seed', 'eval_batch_size', 'target_class',
                      'num_workers', 'jpeg_quality', 'blur_kernel')
    if any(type(getattr(args, key)) is not int for key in integer_fields):
        parser.error('Seed, batch size, class, workers, JPEG quality and blur kernel must be integers')
    if args.eval_seed < 0 or args.eval_batch_size < 1 or args.num_workers < 0:
        parser.error('Invalid seed or data-loader settings')
    if not 0 <= args.target_class < (100 if args.dataset == 'cifar100' else 10):
        parser.error('Invalid target class')
    if not 1 <= args.jpeg_quality <= 100:
        parser.error('JPEG quality must be between 1 and 100')
    if args.blur_kernel < 1 or args.blur_kernel > 31 or args.blur_kernel % 2 == 0:
        parser.error('Blur kernel must be an odd integer between 1 and 31')
    for key in ('embed_strength', 'noise_std'):
        value = getattr(args, key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            parser.error(f'{key} must be a finite nonnegative number')
    return args

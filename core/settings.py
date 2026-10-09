"""Configuration shared by the training and evaluation entry points."""
import argparse
import json
from pathlib import Path


def parse_settings(mode):
    parser = argparse.ArgumentParser(description=f"Table 2 Ours: {mode}")
    parser.add_argument('--config', required=True, help='Dataset JSON configuration')
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--download', action='store_true', help='Allow dataset download')
    if mode == 'test':
        parser.add_argument('--model-dir', required=True)
        parser.add_argument('--results', default=None, help='Optional new JSON output file')
    cli = parser.parse_args()
    config = json.loads(Path(cli.config).read_text(encoding='utf-8'))
    config.pop('_notes', None)
    config.update({k: v for k, v in vars(cli).items() if v is not None})
    args = argparse.Namespace(**config)
    if args.dataset not in ('mnist', 'cifar10', 'cifar100'):
        parser.error('Unsupported dataset')
    if args.arch != 'resnet18':
        parser.error('This release supports ResNet-18 only')
    if not 0 < args.poison_rate <= 1 or args.epochs < 1 or args.warmup_epochs < 0:
        parser.error('Invalid training parameters')
    if args.batch_size < 1 or args.eval_batch_size < 1 or args.num_workers < 0:
        parser.error('Invalid loader parameters')
    if not 0 <= args.target_class < (100 if args.dataset == 'cifar100' else 10):
        parser.error('Invalid target class')
    return args

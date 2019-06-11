#!/usr/bin/env python
import argparse
from function import adaptive_instance_normalization, calc_mean_std
import net
from utils import ImageFolderWithPaths, collate
from torch.utils.data import DataLoader, RandomSampler
from pathlib import Path
from PIL import Image, ImageFile
import random
import torch
import torch.nn as nn
import torchvision.transforms
from torchvision.utils import save_image
from tqdm import tqdm
import os

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated

parser = argparse.ArgumentParser(
    description='This script applies the AdaIN style transfer method to arbitrary datasets.')
parser.add_argument('--content-dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style-dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--output-dir', type=str, default='output',
                    help='Directory to save the output images')
parser.add_argument('--num-styles', type=int, default=1, help='Number of styles to \
                        create for each image (default: 1)')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                          stylization. Should be between 0 and 1')
parser.add_argument('--extensions', nargs='+', type=str, default=['png', 'jpeg', 'jpg'],
                    help='List of image extensions to scan style and content directory for (case sensitive), default: png, jpeg, jpg')

# Advanced options
parser.add_argument('--content-size', type=int, default=0,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style-size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--batch-size', type=int, default=1,
                    help='Number of images to process simultaneously. \
                    Not working unless --content-size , --style-size are set to non-zero values,\
                    and --crop is enabled.')
parser.add_argument('--seed', type=int, default=-1,
                    help='Set random seed, not working if set to negative integers, default: not set')


# random.seed(131213)

def input_transform(size, crop):
    transform_list = []
    if size > 0:
        transform_list.append(torchvision.transforms.Resize(size))
    if crop:
        transform_list.append(torchvision.transforms.CenterCrop(size))
    transform_list.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(transform_list)
    return transform


def content_feat_normalization(content_f):
    size = content_f.data.size()
    content_mean, content_std = calc_mean_std(content_f)
    normalized_feat = (content_f -
                       content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat


def style_transfer(content_f, content_f_norm, style_f, alpha=1.0):
    size = content_f.data.size()
    style_mean, style_std = calc_mean_std(style_f)
    feat = content_f_norm * style_std.expand(size) + style_mean.expand(size)
    transfered_feat = feat * alpha + content_f * (1 - alpha)
    return transfered_feat


def main(args):
    args = parser.parse_args(args)
    assert (0.0 <= args.alpha <= 1.0), 'Alpha should be between 0 and 1'
    if args.seed >= 0:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    # print(args.crop)

    # set content and style directories
    content_dir = Path(args.content_dir)
    style_dir = Path(args.style_dir)
    style_dir = style_dir.resolve()
    output_dir = Path(args.output_dir)
    output_dir = output_dir.resolve()
    assert style_dir.is_dir(), 'Style directory not found'

    # collect content files
    extensions = args.extensions
    assert len(extensions) > 0, 'No file extensions specified'
    content_dir = Path(content_dir)
    content_dir = content_dir.resolve()
    assert content_dir.is_dir(), 'Content directory not found'

    # collect content files
    contents = []
    for ext in extensions:
        contents += list(content_dir.rglob('*.' + ext))

    assert len(contents) > 0, 'No images with specified extensions found in content directory' + content_dir
    content_paths = sorted(contents)
    print('Found %d content images in %s' % (len(content_paths), content_dir))

    # collect style files
    styles = []
    for ext in extensions:
        styles += list(style_dir.rglob('*.' + ext))

    assert len(styles) > 0, 'No images with specified extensions found in style directory' + style_dir
    style_paths = sorted(styles)
    print('Found %d style images in %s' % (len(style_paths), style_dir))

    del contents, styles

    content_tf = input_transform(args.content_size, args.crop)
    style_tf = input_transform(args.style_size, args.crop)

    content_folder = ImageFolderWithPaths(root=args.content_dir,
                                          paths=content_paths,
                                          transform=content_tf)
    style_folder = ImageFolderWithPaths(root=args.style_dir,
                                        paths=style_paths,
                                        transform=style_tf)

    batch_size = args.batch_size if args.content_size > 0 and args.style_size > 0 and args.crop else 1
    content_loader = DataLoader(content_folder, batch_size,
                                num_workers=4,
                                collate_fn=collate)
    style_loader = DataLoader(style_folder, batch_size,
                              sampler=RandomSampler(style_folder, replacement=True),
                              num_workers=4,
                              collate_fn=collate)

    decoder = net.decoder
    vgg = net.vgg

    decoder.load_state_dict(torch.load('models/decoder.pth'))
    vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    decoder.eval()
    vgg.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg.to(device)
    decoder.to(device)
    style_loader_iter = iter(style_loader)

    # actual style transfer as in AdaIN
    with tqdm(total=len(content_paths) * args.num_styles) as pbar:
        for content_imgs, content_paths in content_loader:
            # print(content_paths)
            # save_image(content_imgs, '456.jpg', padding=0)
            content_size = len(content_paths)
            with torch.no_grad():
                content_feat = vgg(content_imgs.to(device))
                content_feat_norm = content_feat_normalization(content_feat)
            for _ in range(args.num_styles):
                style_imgs, style_paths = next(style_loader_iter)
                # print(style_paths)
                if content_size < len(style_paths):
                    # make sure content_size == style_size in every batch
                    style_imgs = style_imgs[:content_size]
                with torch.no_grad():
                    style_feat = vgg(style_imgs.to(device))
                    output = decoder(
                        style_transfer(content_feat, content_feat_norm,
                                       style_feat, args.alpha)
                    ).cpu()
                for idx, (content_path, style_path) in enumerate(zip(content_paths, style_paths)):
                    rel_path = content_path.relative_to(content_dir)
                    out_dir = output_dir.joinpath(rel_path.parent)
                    # create directory structure if it does not exist
                    if not out_dir.is_dir():
                        out_dir.mkdir(parents=True)
                    content_name = content_path.stem
                    style_name = style_path.stem
                    output_name = out_dir.joinpath(f'{content_name}-stylized-{style_name}{content_path.suffix}')
                    save_image(output[idx], output_name, padding=0)
                    # if os.path.isfile(output_name):
                    #     continue
                    pbar.update(1)


if __name__ == '__main__':
    args = None
    # configuration
    # args = ['--content-dir', r'/tmp/train', '--style-dir', r'/tmp/train', '--num-styles', '8',
    #         # '--extensions', 'JPGE',
    #         '--crop',
    #         '--content-size', '128',
    #         '--style-size', '128',
    #         '--batch-size', '8']
    main(args)

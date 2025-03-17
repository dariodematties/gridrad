# This script runs inference on a pre-trained ViT model adapting the hierarchy of the model from ViT16 to ViT256
# i.e. the ViT16 model processes 256x256 images patching them into 16x16 patches, while the ViT256 model processes 2048x2048 images patching them into 8x8 patches
# Each output tensor from the ViT16 model corresponds to a patch of the input image in the higher hierarchy model (ViT256).

import os
import argparse

import utils
from utils import TensorDataset1

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.models as torchvision_models
from torchvision import transforms

import vision_transformer as vits

from einops import rearrange, repeat

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")

    # Misc
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--model_path', default='/path/to/pre-trained/checkpoint/', type=str,
        help='Please specify the path to the pre-trained checkpoint.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save outputs from the inference.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")
    # parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--pretrained_weights", default="", type=str, help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    return parser

def inference_on_pretrained_model(args):
    # Set device and initialize distributed mode
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ loading data ============
    dataset = TensorDataset1(args.data_path)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     sampler=sampler,
    #     batch_size=args.batch_size_per_gpu,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    #     prefetch_factor=2,  # Prefetch batches
    #     persistent_workers=True,  # Keep workers alive for faster batch loading
    # )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,  # Prefetch batches
        persistent_workers=True,  # Keep workers alive for faster batch loading
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            [item[1] for item in batch]
        )
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building model ============
    model = load_model(args)
    print(f"Model loaded: {args.arch} model with patch size {args.patch_size} and output dimension {args.out_dim}.")
    print(f"Model loaded: {model}")

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Inference on 2kx2k images:'
    output_features = []
    output_filenames = []
    for it, (images, filenames) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # move images to device
        images = images.to(args.device, non_blocking=True)

        # forward pass
        with torch.no_grad():
            features_cls256 = forward(images, model, args.device)
            print(f"Features shape: {features_cls256.shape}")
            output_features.append(features_cls256)
            output_filenames.extend(filenames)

    # Concatenate features from local batches
    output_features = concatenate_features(output_features)
    print(f"Local output features shape: {output_features.shape}")

    # ============ Distributed Gathering ============
    # Create a list placeholder to gather tensors from all processes.
    world_size = dist.get_world_size()
    gathered_features = [None for _ in range(world_size)]
    # Use all_gather_object to gather the (possibly varying-sized) tensors
    dist.all_gather_object(gathered_features, output_features)

    # Gather filenames from all processes
    gathered_filenames = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_filenames, output_filenames)

    # Only the main process will save the output features
    if utils.is_main_process():
        # On the main process, concatenate the gathered tensors along the first dimension.
        all_features = torch.cat(gathered_features, dim=0)
        all_filenames = sum(gathered_filenames, []) # Flatten the list of lists
        print(f"All gathered features shape: {all_features.shape}")
        print(f"Total filenames collected: {len(all_filenames)}")

        # Create a dictionary mapping filenames to features
        feature_dict = {
                filename: feature for filename, feature in zip(all_filenames, all_features)
        }

        # Save the output features
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        save_path = os.path.join(args.output_dir, "output_features.pt")
        torch.save(feature_dict, save_path)
        print(f"Saved gathered features to {save_path}")

# def inference_on_pretrained_model(args):
#     args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     utils.init_distributed_mode(args)
#     utils.fix_random_seeds(args.seed)
#     print("git:\n  {}\n".format(utils.get_sha()))
#     print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
#     cudnn.benchmark = True
#
#     # First, we load the data
#     # dataset = TensorDataset(tensors)
#     dataset = TensorDataset(args.data_path)
#     # dataset = datasets.ImageFolder(args.data_path, transform=transform)
#     sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
#     data_loader = torch.utils.data.DataLoader(
#         dataset,
#         sampler=sampler,
#         batch_size=args.batch_size_per_gpu,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         drop_last=True,
#         prefetch_factor=2,  # Prefetch batches
#         persistent_workers=True,  # Keep workers alive for faster batch loading
#     )
#     print(f"Data loaded: there are {len(dataset)} images.")
#
#
#     # ============ building model network ... ============
#     model = load_model(args)
#     print(f"Model loaded: {args.arch} model with patch size {args.patch_size} and output dimension {args.out_dim}.")
#     print(f"Model loaded: {model}")
#
#
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Inference on 2kx2k images:'
#     output_features = []
#     for it, (images) in enumerate(metric_logger.log_every(data_loader, 10, header)):
#         # move images to gpu
#         images = images.to(args.device, non_blocking=True)
#
#         # forward pass
#         with torch.no_grad():
#             features_cls256 = forward(images, model, args.device)
#             print(f"Features shape: {features_cls256.shape}")
#             output_features.append(features_cls256)
#
#     output_features = concatenate_features(output_features)
#     print(f"Output features shape: {output_features.shape}")
#
#     # Save the output features
#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)
#     torch.save(output_features, os.path.join(args.output_dir, "output_features.pt"))

def load_model(args):
    # build model
    model = vits.__dict__[args.arch](
        patch_size=args.patch_size, num_classes=0
    )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(args.device)

    try: 
        os.path.isfile(args.pretrained_weights)
        state_dict = torch.load(args.pretrained_weights, map_location="cpu", weights_only=False)
        if (
            args.checkpoint_key is not None
            and args.checkpoint_key in state_dict
        ):
            print(
                f"Take key {args.checkpoint_key} in provided checkpoint dict"
            )
            state_dict = state_dict[args.checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                args.pretrained_weights, msg
            )
        )
    except FileNotFoundError:
        print("Pretrained model not found in `--pretrained_weights` path")

    return model

def forward(x, model256, device256):
    """
    Forward pass of HIPT (given an image tensor x), outputing the [CLS] token from ViT_4K.
    1. x is center-cropped such that the W / H is divisible by the patch token size in ViT_4K.
    2. x then gets unfolded into a "batch" of [256 x 256] images.
    3. A pretrained ViT_256-16 model extracts the CLS token from each [256 x 256] image in the batch.
    4. These batch-of-features are then reshaped into a 2D features grid (of width "w_256" and height "h_256".)
    5. This feature grid is then used as the input to ViT_4K-256, outputing [CLS]_4K.

    Args:
        - x (torch.Tensor): [1 x C x W' x H'] image tensor.
    Return:
        - features_cls4k (torch.Tensor): [1 x 192] cls token (d_4k = 192 by default).
    """
    # 1. [1 x C x W x H].
    batch_256, w_256, h_256, batch_size, _ = prepare_img_tensor(x)
    # 2. [1 x C x w_256 x h_256 x 256 x 256]
    batch_256 = batch_256.unfold(2, 256, 256).unfold(3, 256, 256)
    # 2. [B x C x 256 x 256], where B = (1*w_256*h_256)
    batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')

    features_cls256 = []
    # 3. B may be too large for ViT_256. We further take minibatch
    for mini_bs in range(0, batch_256.shape[0], 256):
        minibatch_256 = batch_256[mini_bs:mini_bs+256].to(device256, non_blocking=True)
        # 3. Extracting ViT_256 features from [256 x 3 x 256 x 256] image batches.
        features_cls256.append(model256(minibatch_256).detach().cpu())
    

    # 3. [B x 384], where 384 == dim of ViT_256 [CLS] token
    features_cls256 = torch.vstack(features_cls256)
    features_cls256 = features_cls256.reshape(batch_size, w_256, h_256, 384)
    # 4. [B x 384 x w_256 x h_256]
    features_cls256 = features_cls256.to('cpu')
    
    return features_cls256

def prepare_img_tensor(img: torch.Tensor, patch_size=256):
    """
    Helper function that takes a non-square image tensor, and takes a center crop s.t.
    the width / height are divisible by 256.

    (Note: "_256" for w / h should technicaly be renamed as "_ps",
    but may not be easier to read.
    Until I need to make HIPT with patch_sizes != 256,
    keeping the naming convention as-is.)

    Args:
        - img (torch.Tensor): [1 x C x W' x H'] image tensor.
        - patch_size (int): Desired patch size to evenly subdivide the image.
    Return:
        - img_new (torch.Tensor): [1 x C x W x H] image tensor, where W and H
        are divisable by patch_size.
        - w_256 (int): # of [256 x 256] patches of img_new's width (e.g. - W/256)
        - h_256 (int): # of [256 x 256] patches of img_new's height (e.g. - H/256)
    """
    make_divisible = lambda l, patch_size: (l - (l % patch_size))
    b, c, w, h = img.shape
    load_size = make_divisible(w, patch_size), make_divisible(h, patch_size)
    w_256, h_256 = w // patch_size, h // patch_size
    batch_size = b
    num_channels = c
    img_new = transforms.CenterCrop(load_size)(img)

    return img_new, w_256, h_256, batch_size, num_channels

def concatenate_features(feature_list):
    """
    Concatenate a list of feature tensors along the first dimension.
    Each tensor has shape [64, 8, 8, 384]
    
    Args:
        feature_list: List of tensors to concatenate
        
    Returns:
        Concatenated tensor with shape [N*64, 8, 8, 384] where N is len(feature_list)
    """
    return torch.cat(feature_list, dim=0)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    inference_on_pretrained_model(args)

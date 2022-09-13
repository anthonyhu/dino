from glob import glob
import os
import argparse
import numpy as np
from tqdm import tqdm
from torchvision import models as torchvision_models
import torch.nn as nn

from eval_video_segmentation import read_frame_list, read_seg, eval_video_tracking_davis
import utils
import vision_transformer as vits


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with video segmentation on I-PACE data')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'resnet50'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--output_dir', default=".", help='Path where to save segmentations')
    parser.add_argument('--data_path', default='/path/to/davis/', type=str)
    parser.add_argument("--n_last_frames", type=int, default=7, help="number of preceeding frames")
    parser.add_argument("--size_mask_neighborhood", default=12, type=int,
        help="We restrict the set of source nodes considered to a spatial neighborhood of the query node")
    parser.add_argument("--topk", type=int, default=5, help="accumulate label from top k neighbors")
    parser.add_argument("--bs", type=int, default=6, help="Batch size, try to reduce if OOM")
    parser.add_argument("--refresh_label_interval", type=int, default=50, help="Interval to refresh segmentation "
                                                                               "label. "
                                                                      "Default to 50, i.e. 2 seconds.")
    args = parser.parse_args()

    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    # building network
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)

        # imagenet weights
        # import timm
        # model_timm = timm.create_model('vit_small_patch16_224', pretrained=True)
        # model_timm.head = nn.Identity()
        # model.load_state_dict(model_timm.state_dict())
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")

    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        # imagenet weights
        # import torchvision
        # model = torchvision.models.resnet50(pretrained=True)
        # print('resnet50')
        model.fc = nn.Identity()

    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    colour_palette = np.array([
        (220, 20, 60),
        (255, 0, 0),
        (165, 42, 42),
        (0, 0, 142),
        (0, 0, 142),
        (0, 60, 100),
        (0, 0, 230),
        (119, 11, 32),
        (220, 220, 0),
        (250, 170, 30),
        (128, 64, 128),
        (244, 35, 232),
        (196, 196, 196),
        (250, 170, 160),
        (170, 170, 170),
        (140, 140, 200),
        (255, 255, 255),
        (128, 64, 255),
        (70, 70, 70),
        (197, 231, 158),
        (190, 153, 153),
        (255, 255, 128),
        (107, 142, 35),
        (64, 170, 64),
        (70, 130, 180),
        (78, 99, 171),
        (0, 0, 0),
    ])

    # a colour palette must contain 256 colours so fill with dummies.
    colour_palette = np.vstack([colour_palette, np.tile(np.arange(27, 256)[:, None], 3)]).astype(np.uint8)

    list_video_dir = sorted(glob(os.path.join(args.data_path, '*')))
    for i, video_dir in enumerate(list_video_dir):
        video_name = os.path.basename(video_dir)
        print(f'[{i+1}/{len(list_video_dir)}] Segmenting {video_name}')
        frame_list = read_frame_list(os.path.join(video_dir, 'image'))
        for interval in tqdm(range(0, len(frame_list), args.refresh_label_interval)):
            current_frame_list = frame_list[interval:interval+args.refresh_label_interval]
            seg_path = current_frame_list[0].replace("image", "segmentation").replace("jpg", "png")
            first_seg, seg_ori = read_seg(seg_path, args.patch_size)
            eval_video_tracking_davis(args, model, current_frame_list, video_dir, first_seg, seg_ori, colour_palette)
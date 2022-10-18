import argparse
import os
from glob import glob
from PIL import Image, ImageDraw, ImageFont
from torchmetrics import IoU

import numpy as np
import torch
from tqdm import tqdm




WAYVE_COLORMAP = (
    np.array(
        [
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
        ]
    )
    / 255.0
)


def render_semseg(semseg, colormap=WAYVE_COLORMAP):
    img = np.zeros((semseg.shape[0], semseg.shape[1], 3))
    for ind, color in enumerate(colormap):
        img[semseg == ind, :] = color
    return img


def add_legend(img, text='hello', position=(0, 0), colour=[255, 255, 255], size=14):
    font = ImageFont.truetype("DejaVuSans.ttf", size)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, tuple(colour), font=font)
    return np.array(pil_img)


class SegmentationMetric(IoU):
    def update(self, pred, target):
        mask = (target != 255) & (pred != 255)
        if mask.any():
            super().update(pred[mask], target[mask])


def main(args):
    iou_metrics = SegmentationMetric(num_classes=27).cuda()
    for image_path in tqdm(sorted(glob(os.path.join(args.dataset_path, 'image', '*.jpg')))):
        img = Image.open(image_path)
        img = np.array(img)

        segmentation = np.array(Image.open(image_path.replace('image/', 'segmentation/').replace('.jpg', '.png')))
        predicted_path = image_path.replace(os.path.dirname(args.dataset_path), args.prediction_path)
        predicted_path = predicted_path.replace('image/', 'prediction/').replace('.jpg', '.png')
        pred_segmentation = np.array(Image.open(predicted_path))

        if args.compute_metrics:
            segmentation = torch.from_numpy(segmentation).long().cuda()
            pred_segmentation = torch.from_numpy(pred_segmentation).long().cuda()
            iou_metrics.update(pred_segmentation, segmentation)

        if args.save_vis:
            rendered_segmentation = render_semseg(segmentation.cpu().numpy())
            rendered_pred_segmentation = render_semseg(pred_segmentation.cpu().numpy())

            joint_plot = img / 255 * 0.5 + 0.5 * rendered_segmentation
            joint_pred_plot = img / 255 * 0.5 + 0.5 * rendered_pred_segmentation

            vis_path = os.path.dirname(predicted_path).replace('/prediction', '/vis')
            os.makedirs(vis_path, exist_ok=True)
            vis_img = np.hstack([joint_plot, np.zeros((joint_plot.shape[0], 10, 3), dtype=np.float64), joint_pred_plot])
            vis_img = (255 * vis_img).astype(np.uint8)
            vis_img = add_legend(vis_img, 'Ground truth', position=(joint_plot.shape[1] // 2 - 70, 290), size=20)
            vis_img = add_legend(vis_img, 'Prediction',
                                 position=(joint_plot.shape[1] + joint_plot.shape[1] // 2 - 40, 290), size=20)

            Image.fromarray(vis_img).save(os.path.join(vis_path, os.path.basename(image_path).replace('.jpg', '.png')))

    if args.compute_metrics:
        print(f'IoU: {iou_metrics.compute().item()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Calculate segmentation metrics on I-PACE data')
    parser.add_argument('--dataset_path', default='', type=str, help="Path to dataset.")
    parser.add_argument('--prediction_path', default='', type=str, help="Path to the predicted segmentations.")
    parser.add_argument("--save_vis", type=bool, default=True, help="Whether to save overlayed visualisations.")
    parser.add_argument("--compute_metrics", type=bool, default=True, help="Whether to compute metrics.")
    args = parser.parse_args()
    main(args)

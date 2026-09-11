from __future__ import annotations

import math
from typing import Dict

import torch


class SegMetric:
    """Semantic-segmentation metrics based on a confusion matrix.

    The confusion matrix is accumulated with ``torch.bincount``.  When logits
    are passed in, ``argmax`` is computed before any CPU transfer, so training
    does not copy a full ``[B, C, H, W]`` tensor to NumPy just for metrics.
    """

    _ID2NAME: Dict[str, Dict[int, str]] = {
        'carla': {
            0: 'building',
            1: 'fence',
            2: 'trash_can',
            3: 'person',
            4: 'pole',
            5: 'lane_line',
            6: 'road',
            7: 'sidewalk',
            8: 'vegetation',
            9: 'vehicle',
            10: 'wall',
            11: 'traffic_sign',
        },
        'dsec': {
            0: 'sky',
            1: 'building',
            2: 'fence',
            3: 'person',
            4: 'pole',
            5: 'road',
            6: 'sidewalk',
            7: 'vegetation',
            8: 'vehicle',
            9: 'wall',
            10: 'traffic_sign',
        },
        'ddd17': {
            0: 'flat',
            1: 'construction+sky',
            2: 'object',
            3: 'nature',
            4: 'human',
            5: 'vehicle',
        },
    }

    def __init__(self, data_name='carla', ignore_index=255, device=None):
        data_name = str(data_name).lower()
        if data_name not in self._ID2NAME:
            raise ValueError(f"Unknown data name '{data_name}'")

        self.data_name = data_name
        self.id2name = self._ID2NAME[data_name]
        self.ignore_index = int(ignore_index)

        self.valid_ids = sorted(self.id2name.keys())
        self.num_classes = len(self.valid_ids)
        if self.valid_ids != list(range(self.num_classes)):
            raise ValueError(
                'SegMetric currently expects contiguous train IDs starting at 0. '
                f'Got valid_ids={self.valid_ids}.'
            )

        self.reset(device=device)

    def reset(self, device=None):
        """Reset the accumulated confusion matrix."""
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
            device=device,
        )

    def _ensure_device(self, device):
        if self.confusion_matrix.device != device:
            self.confusion_matrix = self.confusion_matrix.to(device=device)

    @staticmethod
    def _as_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.detach()
        return torch.as_tensor(x)

    def update(self, preds, targets):
        """Update the confusion matrix.

        Args:
            preds: logits ``[B, C, H, W]`` or label map ``[B, H, W]`` / ``[H, W]``.
            targets: label map ``[B, H, W]`` / ``[H, W]``.
        """
        with torch.no_grad():
            preds = self._as_tensor(preds)
            targets = self._as_tensor(targets)

            # Compute argmax on the original device before reducing to labels.
            if preds.ndim == 4:
                preds = preds.argmax(dim=1)
            elif preds.ndim == 2:
                preds = preds.unsqueeze(0)
            elif preds.ndim != 3:
                raise ValueError(
                    f'preds shape {tuple(preds.shape)} not supported. '
                    'Expected [B, H, W], [H, W] or logits [B, C, H, W].'
                )

            if targets.ndim == 2:
                targets = targets.unsqueeze(0)
            elif targets.ndim != 3:
                raise ValueError(
                    f'targets shape {tuple(targets.shape)} not supported. '
                    'Expected [B, H, W] or [H, W].'
                )

            if preds.shape != targets.shape:
                raise ValueError(
                    f'pred/target spatial shape mismatch: '
                    f'preds={tuple(preds.shape)}, targets={tuple(targets.shape)}.'
                )

            device = preds.device
            targets = targets.to(device=device, dtype=torch.int64)
            preds = preds.to(device=device, dtype=torch.int64)

            targets = targets.reshape(-1)
            preds = preds.reshape(-1)

            valid = targets != self.ignore_index
            valid &= (targets >= 0) & (targets < self.num_classes)
            valid &= (preds >= 0) & (preds < self.num_classes)
            if not torch.any(valid):
                return

            hist_indices = self.num_classes * targets[valid] + preds[valid]
            hist = torch.bincount(
                hist_indices,
                minlength=self.num_classes ** 2,
            ).reshape(self.num_classes, self.num_classes)

            self._ensure_device(hist.device)
            self.confusion_matrix += hist.to(dtype=self.confusion_matrix.dtype)

    def get_results(self):
        """Return IoU, mIoU, pixel accuracy and mean class accuracy."""
        hist = self.confusion_matrix.to(dtype=torch.float64)

        # Rows are GT classes, columns are predicted classes.
        true_positive = torch.diag(hist)
        gt_area = hist.sum(dim=1)
        pred_area = hist.sum(dim=0)
        union = gt_area + pred_area - true_positive

        iou_per_class = torch.full(
            (self.num_classes,),
            float('nan'),
            dtype=torch.float64,
            device=hist.device,
        )
        valid_iou = union > 0
        iou_per_class[valid_iou] = true_positive[valid_iou] / union[valid_iou]

        acc_per_class = torch.full_like(iou_per_class, float('nan'))
        valid_acc = gt_area > 0
        acc_per_class[valid_acc] = true_positive[valid_acc] / gt_area[valid_acc]

        total_valid_pixels = int(hist.sum().item())
        pixel_acc = (
            float((true_positive.sum() / hist.sum()).item())
            if total_valid_pixels > 0
            else 0.0
        )
        has_valid_iou = bool(torch.any(valid_iou).item())
        has_valid_acc = bool(torch.any(valid_acc).item())
        miou = float(iou_per_class[valid_iou].mean().item()) if has_valid_iou else 0.0
        mean_acc = float(acc_per_class[valid_acc].mean().item()) if has_valid_acc else 0.0

        iou_list = [float(x) for x in iou_per_class.detach().cpu().tolist()]
        acc_list = [float(x) for x in acc_per_class.detach().cpu().tolist()]

        iou_dict = {
            self.id2name[class_id]: iou_list[idx]
            for idx, class_id in enumerate(self.valid_ids)
        }
        acc_dict = {
            self.id2name[class_id]: acc_list[idx]
            for idx, class_id in enumerate(self.valid_ids)
        }
        valid_classes = [
            self.id2name[class_id]
            for idx, class_id in enumerate(self.valid_ids)
            if bool(valid_iou[idx].item())
        ]

        return {
            'iou_per_class': iou_dict,
            'acc_per_class': acc_dict,
            'miou': miou,
            'pixel_acc': pixel_acc,
            'mean_acc': mean_acc,
            'class_iou_list': iou_list,
            'class_acc_list': acc_list,
            'valid_classes': valid_classes,
            'num_valid_classes': int(valid_iou.sum().item()),
            'num_valid_pixels': total_valid_pixels,
        }

    def print_results(self):
        """Print formatted metric results."""
        results = self.get_results()
        label_width = max(
            len(name) for name in list(results['iou_per_class'].keys()) + [
                'mIoU',
                'Pixel Acc',
                'Mean Acc',
            ]
        ) + 2
        print('\n=== Segmentation Metrics ===')
        for name, iou in results['iou_per_class'].items():
            if math.isnan(iou):
                print(f'{name:<{label_width}}: IoU N/A')
            else:
                print(f'{name:<{label_width}}: IoU {iou:.4f}')
        print(
            f"{'mIoU':<{label_width}}: {results['miou']:.4f} "
            f"({results['num_valid_classes']}/{self.num_classes} valid classes)"
        )
        print(f"{'Pixel Acc':<{label_width}}: {results['pixel_acc']:.4f}")
        print(f"{'Mean Acc':<{label_width}}: {results['mean_acc']:.4f}\n")

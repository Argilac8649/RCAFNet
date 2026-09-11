import torch
import random
import torch.nn.functional as F
from torch.utils.data import Dataset


class NormalizedMapDataset(Dataset):
    """
    只做归一化、不做随机增强的数据集包装器。

    用于验证集/测试集，保证其与训练集一样经过 normalize，
    但不会执行随机翻转、模糊等数据增强。
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, event, labels = self.dataset[index]
        image, event = self._normalize(image, event)
        return image, event, labels

    def _normalize(self, image, event):
        if not hasattr(self.dataset, 'normalize'):
            raise AttributeError(
                f"{self.dataset.__class__.__name__} must implement "
                "`normalize(image, event)` when wrapped by NormalizedMapDataset."
            )
        return self.dataset.normalize(image, event)


class AugmentedMapDataset(Dataset):
    """
    训练集数据包装器：先执行数据增强，最后执行归一化。

    注意：
        原始 dataset 应返回未归一化的 image/event。
        随机增强完成后再执行 normalize，避免破坏 normalize 后的数据分布。
    
    参数:
        dataset: 原始数据集对象
        aug_hflip_prob: 随机水平翻转概率
        aug_scale_crop_prob: Random Scale + Crop 概率
    """
    
    def __init__(self,
                 dataset,
                 aug_hflip_prob=0.5,
                 aug_scale_crop_prob=0.3,
                 scale_range=(0.75, 1.5),
                 ignore_index=255):
        self.dataset = dataset
        self.aug_hflip_prob = aug_hflip_prob
        self.aug_scale_crop_prob = aug_scale_crop_prob
        self.scale_range = tuple(scale_range)
        self.ignore_index = ignore_index
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # 从原始数据集获取样本
        image, event, labels = self.dataset[index]

        # 随机尺度缩放 + 裁剪/填充。
        # 这是几何增强，必须同步作用于 image/event/labels。
        if random.random() < self.aug_scale_crop_prob:
            image, event, labels = self.random_scale_crop(
                image, event, labels,
                scale_range=self.scale_range,
                ignore_index=self.ignore_index,
            )
        
        # 随机决定是否应用水平翻转
        if random.random() < self.aug_hflip_prob:
            # 应用水平翻转
            image, event, labels = self.random_hflip(image, event, labels)
        
        # 不启用竖直翻转：对于驾驶场景语义分割，上下翻转会产生非常不自然的样本。
        
        # 数据增强完成后再归一化。
        image, event = self._normalize(image, event)

        # 返回所有样本元素
        return image, event, labels

    def _normalize(self, image, event):
        if not hasattr(self.dataset, 'normalize'):
            raise AttributeError(
                f"{self.dataset.__class__.__name__} must implement "
                "`normalize(image, event)` when wrapped by AugmentedMapDataset."
            )
        return self.dataset.normalize(image, event)
    
    @staticmethod
    def random_hflip(image, event, labels):
        """
        应用水平翻转增强
        
        参数:
            image: 图像张量 [C, H, W]
            event: 事件数据张量 [C, H, W] 
            labels: 标签张量 [H, W]
        
        返回:
            翻转后的张量
        """
        # 水平翻转所有张量（沿宽度维度）
        image = torch.flip(image, dims=[-1])
        event = torch.flip(event, dims=[-1])
        labels = torch.flip(labels, dims=[-1])
        
        return image, event, labels
    
    @staticmethod
    def random_vflip(image, event, labels):
        """
        应用垂直翻转增强
        
        参数:
            image: 图像张量 [C, H, W]
            event: 事件数据张量 [C, H, W] 
            labels: 标签张量 [H, W]
        
        返回:
            翻转后的张量
        """
        # 垂直翻转所有张量（沿高度维度）
        image = torch.flip(image, dims=[-2])
        event = torch.flip(event, dims=[-2])
        labels = torch.flip(labels, dims=[-2])
        
        return image, event, labels
    
    @staticmethod
    def random_crop(image, event, labels, crop_size=(256, 256)):
        """
        随机裁剪增强
        
        参数:
            image: 图像张量 [C, H, W]
            event: 事件数据张量 [C, H, W] 
            labels: 标签张量 [H, W]
            crop_size: 裁剪尺寸 (height, width)
        
        返回:
            裁剪后的张量
        """
        _, H, W = image.shape
        crop_h, crop_w = crop_size
        
        # 计算随机裁剪的起始位置
        top = torch.randint(0, H - crop_h + 1, (1,)).item() if H > crop_h else 0
        left = torch.randint(0, W - crop_w + 1, (1,)).item() if W > crop_w else 0
        
        # 执行裁剪
        image_cropped = image[:, top:top+crop_h, left:left+crop_w]
        event_cropped = event[:, top:top+crop_h, left:left+crop_w]
        labels_cropped = labels[top:top+crop_h, left:left+crop_w]
        
        return image_cropped, event_cropped, labels_cropped

    @staticmethod
    def random_scale_crop(image, event, labels, scale_range=(0.75, 1.5),
                          ignore_index=255):
        """
        随机尺度缩放 + 裁剪/填充，输出尺寸保持与输入一致。

        推荐参数:
            scale_range=(0.75, 1.5)

        说明:
            - image/event 使用 bilinear 插值
            - labels 使用 nearest 插值
            - 当缩放后尺寸小于目标尺寸时，label 用 ignore_index 填充
        """
        _, target_h, target_w = image.shape
        scale = random.uniform(*scale_range)
        new_h = max(1, int(round(target_h * scale)))
        new_w = max(1, int(round(target_w * scale)))

        image = F.interpolate(
            image.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False,
        )[0]
        event = F.interpolate(
            event.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False,
        )[0]
        labels = F.interpolate(
            labels.float().unsqueeze(0).unsqueeze(0),
            size=(new_h, new_w),
            mode='nearest',
        )[0, 0].long()

        # 如果缩放后小于目标尺寸，先填充到至少目标尺寸。
        pad_h = max(target_h - new_h, 0)
        pad_w = max(target_w - new_w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            pad = (pad_left, pad_right, pad_top, pad_bottom)
            image = F.pad(image, pad, value=0.0)
            event = F.pad(event, pad, value=0.0)
            labels = F.pad(labels, pad, value=ignore_index)

        _, h, w = image.shape
        top = random.randint(0, h - target_h) if h > target_h else 0
        left = random.randint(0, w - target_w) if w > target_w else 0

        image = image[:, top:top + target_h, left:left + target_w]
        event = event[:, top:top + target_h, left:left + target_w]
        labels = labels[top:top + target_h, left:left + target_w]

        return image, event, labels

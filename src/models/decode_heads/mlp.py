import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Linear Embedding: 
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class DecoderHead(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):
        
        super(DecoderHead, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        
        self.in_channels = in_channels
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = nn.Identity()

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        
        self.linear_fuse = nn.Sequential(
                            nn.Conv2d(in_channels=embedding_dim*4, out_channels=embedding_dim, kernel_size=1),
                            norm_layer(embedding_dim),
                            nn.ReLU(inplace=True)
                            )
                            
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
       
    def forward(self, inputs):
        # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs
        
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


class ConvBNAct(nn.Module):
    """Lightweight Conv -> Norm -> ReLU block for decoder refinement."""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=padding, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable ConvBNReLU used by the enhanced MLP decoder."""

    def __init__(self, in_channels, out_channels=None, kernel_size=3,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        out_channels = out_channels or in_channels
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size,
                      padding=padding, groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MLPConvHead(nn.Module):
    """
    Enhanced SegFormer-style MLP decoder.

    It keeps the original MLP multi-scale projection/fusion path, then adds a
    lightweight local refinement stage.  The refinement uses depthwise separable
    convolutions by default, which improves local boundary/detail modeling with
    much smaller cost than stacked standard 3x3 convolutions.
    """

    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=512,
                 refine_channels=None,
                 refine_blocks=2,
                 use_depthwise=True,
                 align_corners=False):
        super().__init__()
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        refine_channels = refine_channels or embed_dim

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embed_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embed_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embed_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embed_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, refine_channels, kernel_size=1, bias=False),
            norm_layer(refine_channels),
            nn.ReLU(inplace=True),
        )

        refine_block = DepthwiseSeparableConv if use_depthwise else ConvBNAct
        self.refine = nn.Sequential(*[
            refine_block(refine_channels, refine_channels, 3, norm_layer)
            for _ in range(int(refine_blocks))
        ]) if refine_blocks > 0 else nn.Identity()
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.linear_pred = nn.Conv2d(refine_channels, self.num_classes, kernel_size=1)

    def _linear_project(self, feature, projector, output_size):
        batch = feature.shape[0]
        projected = projector(feature)
        projected = projected.permute(0, 2, 1).reshape(
            batch, -1, feature.shape[2], feature.shape[3]
        )
        if projected.shape[2:] != output_size:
            projected = F.interpolate(
                projected,
                size=output_size,
                mode='bilinear',
                align_corners=self.align_corners,
            )
        return projected

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        output_size = c1.shape[2:]

        c4 = self._linear_project(c4, self.linear_c4, output_size)
        c3 = self._linear_project(c3, self.linear_c3, output_size)
        c2 = self._linear_project(c2, self.linear_c2, output_size)
        c1 = self._linear_project(c1, self.linear_c1, output_size)

        x = self.linear_fuse(torch.cat([c4, c3, c2, c1], dim=1))
        x = self.refine(x)
        x = self.dropout(x)
        return self.linear_pred(x)


if __name__ == '__main__':
    batch_size = 2
    in_channels=[32, 64, 160, 256]

    # 模拟两个尺度的特征（必须与 feat_shapes 一致）
    features = [
        torch.rand(batch_size, in_channels[0], 57, 100),
        torch.rand(batch_size, in_channels[1], 29, 50),   
        torch.rand(batch_size, in_channels[2], 15, 25),  
        torch.rand(batch_size, in_channels[3], 8, 13),
    ]

    model = DecoderHead(in_channels)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f}M")  # 0.51M

    with torch.no_grad():
        output = model(features)
        print("输出形状:", output.shape)  # torch.Size([2, 40, 57, 100]) 模型总参数量: 2.79M

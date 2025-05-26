import math
import torch
from torch import nn

from pysot.utils.misc1 import NestedTensor
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor,mask):
        x = tensor
        mask = mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)   # .cumsum是cumulative sum累积求和操作，1为按行累计求和，如第一行累计求和的值赋给第一行，第二行的值为前两行累计求和。
        x_embed = not_mask.cumsum(2, dtype=torch.float32)   # 2按列累计求和 ，求出了t值
        if self.normalize:  # 对编码进行归一化后乘以一个系数
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)  # self.num_pos_feats向量维度 也就是i
        dim_t = self.temperature ** (2 * torch.div(dim_t,2,rounding_mode='floor') / self.num_pos_feats) # 频率减小，振幅变大。
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3) # pos_x相邻位置sin，cos编码
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
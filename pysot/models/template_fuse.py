import copy
from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor

class TMfuse(nn.Module):
    def __init__(self,d_model,nhead,dim_feedforward,
                 num_featurefusion_layers,dropout=0.1,activation='relu'):
        super().__init__()
        
        featurefusion_layer = FeatureFusionLayer(d_model, nhead, dim_feedforward, dropout, activation)  # 模板特征融合层
        self.fusion = Fusion(featurefusion_layer, num_featurefusion_layers)   # 2层featurefusion_layer
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # 均匀分布，初始化权重xavier_uniform_为了输入与输出方差近似
    
    def forward(self, src_template,src_dny_template,mask_tem,mask_dny_tem,pos_tem,pos_dny_tem):
        b,c,h,w = src_template.size()
        src1 = src_template.flatten(2).permute(2,0,1).contiguous()
        src2 = src_dny_template.flatten(2).permute(2,0,1).contiguous()
        pos_src1 = pos_tem.flatten(2).permute(2,0,1).contiguous()
        pos_src2 = pos_dny_tem.flatten(2).permute(2,0,1).contiguous()
        src1_mask = mask_tem.flatten(1)
        src2_mask = mask_dny_tem.flatten(1)

        # output = self.fusion(src1,src2,
        #                     src1_key_padding_mask=src1_mask,
        #                     src2_key_padding_mask=src2_mask,
        #                     pos_src1=pos_src1,pos_src2=pos_src2)
        
        output = self.fusion(src2,src1,
                            src1_key_padding_mask=src2_mask,
                            src2_key_padding_mask=src1_mask,
                            pos_src1=pos_src2,pos_src2=pos_src1)
        
        output = src1 + output
        output = self.norm(output)
        return output.view(w,h,b,c).permute(2,3,1,0).contiguous()

        
    
class FeatureFusionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):

        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model, dim_feedforward)
        self.linear12 = nn.Linear(dim_feedforward, d_model)

        self.linear21 = nn.Linear(d_model, dim_feedforward)
        self.linear22 = nn.Linear(dim_feedforward, d_model)

        self.norm11 = nn.LayerNorm(d_model)
        self.norm12 = nn.LayerNorm(d_model)
        self.norm21 = nn.LayerNorm(d_model)
        self.norm22 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)

        self.dropout21 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)


        self.activation1 = _get_activation_fn(activation)
        self.activation2 = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,             # src1 DT src2 T
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):
        
        src = self.multihead_attn1(self.with_pos_embed(src1,pos_src1),
                                   self.with_pos_embed(src2,pos_src2),
                                   src2,attn_mask=src2_mask,
                                   key_padding_mask=src2_key_padding_mask)[0]
        src = src1 + self.dropout11(src)
        src = self.norm11(src)

        srcs1 = self.linear12(self.dropout11(self.activation1(self.linear11(src))))
        src = src + self.dropout12(srcs1)
        src = self.norm12(src)
        
        srcs2 = self.multihead_attn2(src,src,src,attn_mask=src2_mask,
                                   key_padding_mask=src2_key_padding_mask)[0]
        
        src = src + self.dropout21(srcs2)
        src = self.norm21(src)

        srcs2 = self.linear22(self.dropout21(self.activation2(self.linear21(src))))
        src = src + self.dropout22(srcs2)
        src = self.norm22(src)
        
        return src

class Fusion(nn.Module):
    def __init__(self,layer,num,) -> None:
        super().__init__()
        self.layer = _get_clones(layer,num)
        self.num = num
        
    def forward (self, src1, src2,                      # src1 DT src2 T
                 src1_mask: Optional[Tensor] = None,
                 src2_mask: Optional[Tensor] = None,
                 src1_key_padding_mask: Optional[Tensor] = None,
                 src2_key_padding_mask: Optional[Tensor] = None,
                 pos_src1: Optional[Tensor] = None,
                 pos_src2: Optional[Tensor] = None):
        output = src2
        for layer in self.layer:
            output = layer(src1,output,
                           src1_mask=src1_mask,src2_mask=src2_mask,
                           src1_key_padding_mask=src1_key_padding_mask,
                           src2_key_padding_mask=src2_key_padding_mask,
                           pos_src1=pos_src1,pos_src2=pos_src2)
        return output

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
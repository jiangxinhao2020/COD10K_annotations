# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, xavier_init
from mmcv.runner import BaseModule, ModuleList

from ..builder import NECKS, build_backbone
from .fpn import FPN

from torch.nn import Softmax

 
 
class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.softmax = Softmax(dim=1)
        self.conv11 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True))      
        self.bn1 = nn.BatchNorm2d(self.channels, affine=True)
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
        self.bn3 = nn.BatchNorm2d(self.channels, affine=True)
        self.bn4 = nn.BatchNorm2d(self.channels, affine=True)



    def forward(self, x):
        residual = x
        x111 = self.conv11(x)
        b2,c2,h2,w2 = x111.size()
        if b2 == 4:
            x1,x2,x3,x4 = torch.chunk(x111, b2, dim=0)
            x1 = self.bn1(x1)
            weight_bn1 = self.bn1.weight.data.abs() / torch.sum(self.bn1.weight.data.abs())
            x1 = x1.permute(0, 2, 3, 1).contiguous()
            x1 = torch.mul(weight_bn1, x1)
            x1 = x1.permute(0, 3, 1, 2).contiguous()         
            x2 = self.bn2(x2)
            weight_bn2 = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
            x2 = x2.permute(0, 2, 3, 1).contiguous()
            x2 = torch.mul(weight_bn2, x2)
            x2 = x2.permute(0, 3, 1, 2).contiguous() 
            x3 = self.bn3(x3)
            weight_bn3 = self.bn3.weight.data.abs() / torch.sum(self.bn3.weight.data.abs())
            x3 = x3.permute(0, 2, 3, 1).contiguous()
            x3 = torch.mul(weight_bn3, x3)
            x3 = x3.permute(0, 3, 1, 2).contiguous() 
            x4 = self.bn4(x4)
            weight_bn4 = self.bn4.weight.data.abs() / torch.sum(self.bn4.weight.data.abs())
            x4 = x4.permute(0, 2, 3, 1).contiguous()
            x4 = torch.mul(weight_bn4, x4)
            x4 = x4.permute(0, 3, 1, 2).contiguous() 
            cat = torch.cat((x1, x2, x3, x4), dim=0)
        else:
            x1 = self.bn1(x111)
            weight_bn1 = self.bn1.weight.data.abs() / torch.sum(self.bn1.weight.data.abs())
            x1 = x1.permute(0, 2, 3, 1).contiguous()
            x1 = torch.mul(weight_bn1, x1)
            cat = x1.permute(0, 3, 1, 2).contiguous()
        x = self.softmax(cat) * x111
        
        return x + residual


class SPModule(nn.Module):
    def __init__(self, channels):
        super(SPModule, self).__init__()
        self.Channel_Att = Channel_Att(channels)

  
    def forward(self, x):

        x_out1=self.Channel_Att(x)
 
        return x_out1  




class ASPP(BaseModule):
    """ASPP (Atrous Spatial Pyramid Pooling)

    This is an implementation of the ASPP module used in DetectoRS
    (https://arxiv.org/pdf/2006.02334.pdf)

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by this module
        dilations (tuple[int]): Dilations of the four branches.
            Default: (1, 3, 6, 1)
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilations=(1, 3, 6, 1),
                 init_cfg=dict(type='Kaiming', layer='Conv2d')):
        super().__init__(init_cfg)
        assert dilations[-1] == 1
        self.aspp = nn.ModuleList()
        for dilation in dilations:
            kernel_size = 3 if dilation > 1 else 1
            padding = dilation if dilation > 1 else 0
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                bias=True)
            self.aspp.append(conv)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(len(self.aspp)):
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out



class GCN(nn.Module):
    """
        Implementation of simple GCN operation in Glore Paper cvpr2019
    """
    def __init__(self, node_num, node_fea):
        super(GCN, self).__init__()
        self.node_num = node_num
        self.node_fea = node_fea
        self.conv_adj = nn.Conv1d(self.node_num, self.node_num, kernel_size=1, bias=False)
        self.bn_adj = nn.BatchNorm1d(self.node_num)

        self.conv_wg = nn.Conv1d(self.node_fea, self.node_fea, kernel_size=1, bias=False)
        self.bn_wg = nn.BatchNorm1d(self.node_fea)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
            x: shape: (b, n, d) 4 1 256
        """
        z = self.conv_adj(x)
        z = self.bn_adj(z)
        z = self.relu(z)

        # Laplacian smoothing
        z += x
        z = z.transpose(1, 2).contiguous()  # (b, d, n)
        z = self.conv_wg(z)
        z = self.bn_wg(z)
        z = self.relu(z)
        z = z.transpose(1, 2).contiguous()  # (b, n, d)

        return z


class FPModule(nn.Module):
    def __init__(self,  num_points=256, thresholds=0.8):
        super(FPModule, self).__init__()
        self.num_points = 256
        self.thresholds = 0.8
        self.gcn = GCN(1, 256)
        self.glb = nn.AdaptiveAvgPool2d((1,1))
        self.conv256_3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True))
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        """
            x is body feature (upsampled)
            edge is boundary feature map
            both features are the same size
        """
        x1 = self.conv256_3(x)
        x2 = self.glb(x1) #4 256 1 1

        x2 = x2.permute(0, 2, 3, 1) #4 1 1 256
        x2 = x2.flatten(start_dim=2) #4 1 256
        x3 = self.gcn(x2)  #4 1 256
        x3 = x3.unsqueeze(1)#4 1 1 256
        x3 = x3.permute(0, 3, 2, 1) #4 256 1 1
        x3 = self.softmax(x3)

        x4 = torch.mul(x1, x3)
        final_features = x+x4

        return final_features


class Downblock(nn.Module):
    def __init__(self, channels):
        super(Downblock, self).__init__()
        self.dwconv1 = nn.Conv2d(channels, channels, stride=1,
                                 kernel_size=3, padding=2, dilation=2, bias=False)
        self.dwconv2 = nn.Conv2d(channels, channels, stride=1,
                                 kernel_size=5, padding=4, dilation=2, bias=False)

    def forward(self, x):
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x = torch.add(x1, x2)
        return x


class KPModule(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(KPModule,self).__init__()
        self.softmax = Softmax(dim=1)
        self.conv11 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True))
        self.bnrelu1 = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.dwconvs = Downblock(256)

    def forward(self, x):

        m_batchsize, _, height, width = x.size()
        x1 = self.conv11(x)
        x_h = torch.nn.functional.adaptive_avg_pool2d(x1, (height, 1))
        x_w = torch.nn.functional.adaptive_avg_pool2d(x1, (1, width))
        cheng = torch.matmul(x_h, x_w)
        cheng = self.bnrelu1(cheng)
        cheng = self.conv11(cheng)
        dwc = self.dwconvs(cheng)
        dwc = self.bnrelu1(dwc)
        dwc = self.conv11(dwc)
        concate = self.softmax(dwc)

        hadama =torch.mul(x1, concate)

        return hadama + x



class DyHeadBlock(nn.Module):
    """DyHead Block with three types of attention.

    HSigmoid arguments in default act_cfg follow official code, not paper.
    https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        zero_init_offset (bool, optional): Whether to use zero init for
            `spatial_conv_offset`. Default: True.
        act_cfg (dict, optional): Config dict for the last activation layer of
            scale-aware attention. Default: dict(type='HSigmoid', bias=3.0,
            divisor=6.0).
    """

    def __init__(self, act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)):
        super().__init__()

        self.FPM = FPModule(256)
        self.KPM = KPModule(256)
        self.SPM = SPModule(256)

    def forward(self, x):
        """Forward function."""
        outs = []
        for level in range(len(x)):

            aaa = self.FPM(x[level])
            outs.append(self.KPM(self.SPM(aaa)))

        return outs



@NECKS.register_module()
class RFP(FPN):
    """RFP (Recursive Feature Pyramid)

    This is an implementation of RFP in `DetectoRS
    <https://arxiv.org/pdf/2006.02334.pdf>`_. Different from standard FPN, the
    input of RFP should be multi level features along with origin input image
    of backbone.

    Args:
        rfp_steps (int): Number of unrolled steps of RFP.
        rfp_backbone (dict): Configuration of the backbone for RFP.
        aspp_out_channels (int): Number of output channels of ASPP module.
        aspp_dilations (tuple[int]): Dilation rates of four branches.
            Default: (1, 3, 6, 1)
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 rfp_steps,
                 rfp_backbone,
                 aspp_out_channels,
                 num_blocks=6,
                 aspp_dilations=(1, 3, 6, 1),
                 zero_init_offset=True,
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.up = nn.Upsample(scale_factor=0.5, mode='nearest')
        self.rfp_steps = rfp_steps
        self.dyhead_blocks0519 = DyHeadBlock()
        self.rfp_modules = ModuleList()
        for rfp_idx in range(1, rfp_steps):
            rfp_module = build_backbone(rfp_backbone)
            self.rfp_modules.append(rfp_module)
        self.rfp_aspp = ASPP(self.out_channels, aspp_out_channels,
                             aspp_dilations)
        self.rfp_weight = nn.Conv2d(
            self.out_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        dyhead_blocks = []
        for i in range(num_blocks):
            in_channels = self.in_channels if i == 0 else self.out_channels
            dyhead_blocks.append(
                DyHeadBlock())
        self.dyhead_blocks = nn.Sequential(*dyhead_blocks)
        self.CBR = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True))
        self.dwonsampel = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=2,padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True))

    def init_weights(self):

        for convs in [self.lateral_convs, self.fpn_convs]:
            for m in convs.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')
        for rfp_idx in range(self.rfp_steps - 1):
            self.rfp_modules[rfp_idx].init_weights()
        constant_init(self.rfp_weight, 0)

    def forward(self, inputs):
        
        inputs = list(inputs)
        
        assert len(inputs) == len(self.in_channels) + 1  # +1 for input image
        img = inputs.pop(0)
        # FPN forward
        x = super().forward(tuple(inputs))

        rfp_feats = [x[0]] + list(
            self.rfp_aspp(x[i]) for i in range(1, len(x)))
        a0 = self.CBR(rfp_feats[0])
        a1 = self.CBR(self.dwonsampel(a0) + rfp_feats[1])
        a2 = self.CBR(self.dwonsampel(a1) + rfp_feats[2])
        a3 = self.CBR(self.dwonsampel(a2) + rfp_feats[3])
        a4 = self.CBR(self.dwonsampel(a3) + rfp_feats[4])
 
        aaa = [a0] + [a1] + [a2] + [a3] + [a4]

        x_new = []
        for ft_idx in range(len(aaa)):
            x_new.append(aaa[ft_idx] + x[ft_idx])
                      
        x = self.dyhead_blocks0519( x_new  )

        return x

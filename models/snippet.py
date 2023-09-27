import copy
import copyreg
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import numpy as np

model_urls = {
    'scnet50_v1d': 'https://backseason.oss-cn-beijing.aliyuncs.com/scnet/scnet50_v1d-4109d1e1.pth',
}


class IntraSIM(nn.Module):
    def __init__(self, u, t, ch_in, norm_layer):
        super(IntraSIM, self).__init__()

        self.u = u
        self.t = t

        ch_in_half = ch_in // 2
        self.conv1 = nn.Conv2d(ch_in, ch_in_half, kernel_size=1, bias=False)
        self.bn1 = norm_layer(ch_in_half)
        self.conv2 = nn.Conv2d(ch_in, ch_in_half, kernel_size=1, bias=False)
        self.bn2 = norm_layer(ch_in_half)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(ch_in_half, ch_in_half)
        self.fc2 = nn.Linear(ch_in_half, ch_in_half)

        self.relu = nn.ReLU(inplace=True)

        self.intra_sma = IntraSMA(u, t, ch_in_half, norm_layer)

    def forward(self, x):
        """

        Args:
            u: 片段数量,t:每个片段图片数量
            x: u * c * t * h * w
        Returns:

        """
        # x1,x2 相对于x通道数减半
        i1 = self.relu(self.bn1(self.conv1(x)))
        i2 = self.relu(self.bn2(self.conv2(x)))
        i2_k = i2

        # i2的两条分支
        # x1分支接全连接层
        i2_k = self.avg_pool(i2_k)
        i2_k = i2_k.reshape(i2_k.shape[0], -1)
        i2_k = self.fc1(i2_k)
        i2_k = self.relu(i2_k)

        i2_k = self.fc2(i2_k)
        i2_k = torch.softmax(i2_k, dim=1)

        # x2分支进入sma模块
        x2 = self.intra_sma(i2)
        o2 = i2_k.unsqueeze(-1).unsqueeze(-1) * x2

        return torch.cat([i1, o2], dim=1)


class IntraSMA(nn.Module):
    def __init__(self, u, t, ch_in, norm_layer):
        super(IntraSMA, self).__init__()

        self.u = u
        self.t = t

        # self.reduction=16
        self.reduced_channels = ch_in // 1
        self.conv1 = nn.Conv2d(ch_in, self.reduced_channels, kernel_size=1, padding=0, bias=False)
        self.bn1 = norm_layer(self.reduced_channels)
        self.relu = nn.ReLU(inplace=True)

        self.ch_in2 = self.reduced_channels // 1
        # 这里,应该是每张图片的输入通道
        self.conv2 = nn.Conv2d(self.ch_in2, self.ch_in2, kernel_size=3, padding=1, bias=False)

        # self.conv_ht1 = nn.Conv2d(self.ch_in2, self.ch_in2, kernel_size=3, padding=1, bias=False)
        self.conv_ht1 = nn.Conv2d(self.ch_in2, self.ch_in2,
                                  kernel_size=(3, 1), padding=(1, 0), groups=self.ch_in2, bias=False)
        self.conv_ht2 = nn.Conv2d(self.ch_in2, self.ch_in2, kernel_size=1, padding=0, bias=False)

        # self.conv_wt1 = nn.Conv2d(self.ch_in2, self.ch_in2, kernel_size=3, padding=1, bias=False)
        self.conv_wt1 = nn.Conv2d(self.ch_in2, self.ch_in2,
                                  kernel_size=(3, 1), padding=(1, 0), groups=self.ch_in2, bias=False)
        self.conv_wt2 = nn.Conv2d(self.ch_in2, self.ch_in2, kernel_size=1, padding=0, bias=False)

        # self.conv_ht3 = nn.Conv2d(self.ch_in2, self.ch_in2, kernel_size=1, padding=0, bias=False)
        # self.conv_wt3 = nn.Conv2d(self.ch_in2, self.ch_in2, kernel_size=1, padding=0, bias=False)

    def reshape_feat(self, feat_):
        """

        Args:
            feat: shape=n,c,h,w, n=b*u*t

        Returns:

        """
        feat = feat_.reshape((-1, self.t) + feat_.shape[1:])
        # u分段数 t每段的图片数量
        bu, t, c, h, w = feat.shape

        # 使得每个分段的图片次序在首位
        # t,bu,c,h,w
        feat = feat.permute(1, 0, 2, 3, 4).contiguous()
        t_list = []

        # =====================================================================================================
        for i in range(t):
            if i == t - 1:
                break
            diff_feat = feat[i] - self.conv2(feat[i + 1])
            t_list.append(diff_feat)

        feat_stack = torch.stack(t_list, dim=0)
        t1, bu, c_, h_, w_ = feat_stack.shape
        # bu*w,c,h,t
        diff_h = feat_stack.permute(1, 4, 2, 3, 0).contiguous().reshape(-1, c_, h_, t1)
        diff_h = self.conv_ht2(self.conv_ht1(diff_h) + diff_h)
        # bu*h,c,t,w
        diff_w = feat_stack.permute(1, 3, 2, 0, 4).contiguous().reshape(-1, c_, t1, w_)
        diff_w = self.conv_wt2(self.conv_wt1(diff_w) + diff_w)

        diff_h = torch.sigmoid(torch.mean(diff_h, dim=-1, keepdim=True)).reshape(-1, w_, c_, h_, 1)
        diff_h = diff_h.permute(0, 4, 2, 3, 1).contiguous()

        diff_w = torch.sigmoid(torch.mean(diff_w, dim=-2, keepdim=True)).reshape(-1, h_, c_, 1, w_)
        diff_w = diff_w.permute(0, 3, 2, 1, 4).contiguous()

        # ====================================================================================================
        t_list2 = []
        for i in range(t):
            if i == t - 1:
                break
            diff_feat = feat[i + 1] - self.conv2(feat[i])
            t_list2.append(diff_feat)

        feat_stack2 = torch.stack(t_list, dim=0)
        t1, bu, c_, h_, w_ = feat_stack2.shape
        # bu*w,c,h,t
        diff_h2 = feat_stack2.permute(1, 4, 2, 3, 0).contiguous().reshape(-1, c_, h_, t1)
        diff_h2 = self.conv_ht2(self.conv_ht1(diff_h2) + diff_h2)
        # bu*h,c,t,w
        diff_w2 = feat_stack.permute(1, 3, 2, 0, 4).contiguous().reshape(-1, c_, t1, w_)
        diff_w2 = self.conv_wt2(self.conv_wt1(diff_w2) + diff_w2)

        diff_h2 = torch.sigmoid(torch.mean(diff_h2, dim=-1, keepdim=True)).reshape(-1, w_, c_, h_, 1)
        diff_h2 = diff_h2.permute(0, 4, 2, 3, 1).contiguous()

        diff_w2 = torch.sigmoid(torch.mean(diff_w2, dim=-2, keepdim=True)).reshape(-1, h_, c_, 1, w_)
        diff_w2 = diff_w2.permute(0, 3, 2, 1, 4).contiguous()

        diff_h = (diff_h + diff_h2) / 2
        diff_w = (diff_w + diff_w2) / 2

        # (b*u,1,c,h,w), 1是指每个分段所有特征的平均值
        return diff_h, diff_w

        # a = diff_h * diff_w
        # feat_h = f.permute(0, 4, 2, 3, 1).contiguous().view(-1, c, h, t)
        # diff_h = self.conv_ht2(self.conv_ht1(feat_h) + feat_h)
        #
        # feat_w = f.permute(0, 3, 2, 1, 4).contiguous().view(-1, c, t, w)
        # diff_w = self.conv_wt2(self.conv_wt1(feat_w) + feat_w)

        # diff_h_list, diff_w_list = [], []
        # for f in t_list:
        #     f = f.unsqueeze(-1)
        #     bu, c1, h1, w1, t1 = f.size()
        #     # bu, c1, h1, w1 = f.size()
        #     # [bu, c, h, w, t] -> [bu, w, c, h, t] -> [bnw, c, h, t]  t=1
        #     feat_h = f.permute(0, 3, 1, 2, 4).contiguous().reshape(-1, c1, h1, t1)
        #     # feat_h = f.permute(0, 3, 1, 2).contiguous().reshape(-1, c1, h1)
        #     a = self.conv_ht1(feat_h)
        #     b = a + feat_h
        #     c = self.conv_ht2(b)
        #     diff_h = self.conv_ht2(self.conv_ht1(feat_h) + feat_h)
        #     diff_h = diff_h.reshape(bu, w1, c1, h1, t1).permute(0, 2, 3, 1, 4).contiguous().unsqueeze()
        #     #  -> [bu,h,c,t,w]
        #     # feat_w = f.permute(0, 2, 1, 4, 3).contiguous().reshape(-1, c1, t1, w1)
        #     feat_w = f.permute(0, 2, 1, 3).contiguous().reshape(-1, c1, w1)
        #     diff_w = self.conv_wt2(self.conv_wt1(feat_w) + feat_w)
        #     diff_w = diff_w.reshape(bu, h1, c1, w1).permute(0, 2, 1, 3).contiguous()
        #     # feat_w = f.permute(0, 2, 1, 3).contiguous().reshape(-1, c1, w1)
        #     # diff_w = self.conv_wt2(self.conv_wt1(feat_w) + feat_w)
        #     diff_h_list.append(diff_h)
        #     diff_w_list.append(diff_w)
        #
        # diff_h_avg = sum(diff_h_list) / len(diff_h_list)
        # diff_w_avg = sum(diff_w_list) / len(diff_w_list)
        #
        # diff_h_avg = torch.sigmoid(self.conv_ht3(diff_h_avg))
        # diff_w_avg = torch.sigmoid(self.conv_wt3(diff_w_avg))
        #
        # return diff_h_avg, diff_w_avg

    def forward(self, x2):
        x2 = self.relu(self.bn1(self.conv1(x2)))
        diff_h, diff_w = self.reshape_feat(x2)
        but, c, h, w = x2.shape
        x2_ = x2.reshape(-1, self.t, c, h, w)

        sma1 = diff_h * diff_w * x2_
        sma1 = sma1.reshape(but, c, h, w)

        sma = sma1 + x2

        return sma


class InterSMA(nn.Module):
    def __init__(self, u, t, ch_in, norm_layer):
        super(InterSMA, self).__init__()
        self.u = u
        self.t = t

        ch_in_half = ch_in // 2
        self.conv1 = nn.Conv2d(ch_in, ch_in_half, kernel_size=1, bias=False)
        self.bn1 = norm_layer(ch_in_half)

        # self.conv2 = nn.Conv2d(ch_in_half, ch_in_half, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.conv2 = nn.Conv2d(ch_in_half, ch_in_half, kernel_size=(3, 1), padding=(1, 0), bias=False)
        # self.bn2 = norm_layer(ch_in_half)

        self.conv3 = nn.Conv2d(ch_in_half, ch_in, kernel_size=1, bias=False)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # self.fc1 = nn.Linear(512 * block.expansion, num_classes * 64)
        # self.fc2 = nn.Linear(num_classes * 64, num_classes)

        self.relu = nn.ReLU(inplace=True)

    def reshape_feat(self, feat_):
        """
        Args:
            feat: shape=n,c,h,w, n=b*u*t
        Returns:
        """
        feat = feat_.reshape((-1, self.t) + feat_.shape[1:])
        # u分段数 t每段的图片数量
        # b,u,c,t,1
        # b, u, c, t, n = feat_.shape
        # x2 = feat_.reshape(-1, self.u, c, t, n)
        # =================bt,u,c,1

        # b, u, c, t, n = x2.shape
        # u,b,c,t,n
        x2 = feat_.permute(1, 0, 2, 3, 4).contiguous()
        # bu, c, t, n = x2.shape

        u_list = []
        for i in range(u):
            if i == u - 1:
                break
            diff_feat = x2[i] - self.conv2(x2[i + 1])
            u_list.append(diff_feat)

        diff_avg = sum(u_list) / len(u_list)

        diff_avg = torch.sigmoid(self.conv3(diff_avg))

        return diff_avg

    def reshape_feat2(self, feat_):
        """
        Args:
            feat: shape=n,c,h,w, n=b*u*t
        Returns:
        """
        # b, c, u, t = feat_.shape

        # bu, c, t, 1
        bu, c, t, n = feat_.shape
        feat_ = feat_.reshape(-1, self.u, c, t, n)
        b, u, c, t, n = feat_.shape
        # u,b,c,t,1
        feat_ = feat_.permute(1, 0, 2, 3, 4).contiguous()

        # feat = feat_.reshape((-1, self.t) + feat_.shape[1:])
        # bu,c,t
        # feat_new = feat_.squeeze(-1).reshape(-1, self.u, c, t)
        # b, u, c, t = feat_new.shape
        # b,c,u,t
        # x2 = feat_.permute(2, 0, 1, 3).contiguous()
        # u,b,c,t,1
        # x2 = x2.unsqueeze(-1)

        u_list = []
        for i in range(u):
            if i == u - 1:
                break
            diff_feat = feat_[i] - self.conv2(feat_[i + 1])
            u_list.append(diff_feat)
        # u-1,b,c,t,1
        diff_u = torch.stack(u_list, dim=0)
        # b,c,u,t,1
        diff_u = diff_u.permute(1, 2, 0, 3, 4).contiguous()
        # b,c,u,t
        diff_u = diff_u.squeeze(-1)
        # b,c,1,t
        diff_u = torch.mean(diff_u, dim=-2, keepdim=True)

        u_list2 = []
        for i in range(u):
            if i == u - 1:
                break
            diff_feat2 = feat_[i + 1] - self.conv2(feat_[i])
            u_list2.append(diff_feat2)
        # u-1,b,c,t,1
        diff_u2 = torch.stack(u_list2, dim=0)
        # b,c,u,t,1
        diff_u2 = diff_u2.permute(1, 2, 0, 3, 4).contiguous()
        # b,c,u,t
        diff_u2 = diff_u2.squeeze(-1)
        # b,c,1,t
        diff_u2 = torch.mean(diff_u2, dim=-2, keepdim=True)

        diff_u = (diff_u + diff_u2) / 2

        # b,c,t,1
        # diff_u = sum(u_list) / len(u_list)
        # u-1,b,c,t,1
        # diff_u = torch.stack(u_list, dim=0)
        # diff_avg = torch.sigmoid(self.conv3(diff_avg))

        return diff_u

    def reshape_feat3(self, feat_):
        """
        Args:
            feat: shape=n,c,h,w, n=b*u*t
        Returns:
        """
        b, c, u, t = feat_.shape

        diff_u1 = feat_ - self.self.conv2(f)

        # bu, c, t, 1
        # bu, c, t, n = feat_.shape
        feat_ = feat_.reshape(-1, self.u, c, t, n)
        b, u, c, t, n = feat_.shape
        # u,b,c,t,1
        feat_ = feat_.permute(1, 0, 2, 3, 4).contiguous()

        # feat = feat_.reshape((-1, self.t) + feat_.shape[1:])
        # bu,c,t
        # feat_new = feat_.squeeze(-1).reshape(-1, self.u, c, t)
        # b, u, c, t = feat_new.shape
        # b,c,u,t
        # x2 = feat_.permute(2, 0, 1, 3).contiguous()
        # u,b,c,t,1
        # x2 = x2.unsqueeze(-1)

        u_list = []
        for i in range(u):
            if i == u - 1:
                break
            diff_feat = feat_[i] - self.conv2(feat_[i + 1])
            u_list.append(diff_feat)
        # b,c,t,1
        diff_u = sum(u_list) / len(u_list)
        # u-1,b,c,t,1
        # diff_u = torch.stack(u_list, dim=0)
        # diff_avg = torch.sigmoid(self.conv3(diff_avg))

        return diff_u

    def forward(self, x2):
        # bu,c,t,1
        # b,c,u,t
        x2 = self.relu(self.bn1(self.conv1(x2)))
        # 通道变为c//4
        # diff_avg = self.reshape_feat(x2)
        diff_u = self.reshape_feat2(x2)
        # sma = diff_w * diff_w * x2 + x2
        diff_u = torch.sigmoid(self.conv3(diff_u))
        # return diff_avg
        # b, c, 1, t
        return diff_u


class InterSIM(nn.Module):
    def __init__(self, u, t, ch_in, norm_layer):
        self.u = u
        self.t = t
        super(InterSIM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reduced_channels = ch_in // 1

        self.conv3x1 = nn.Conv2d(ch_in, self.reduced_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(self.reduced_channels)

        self.conv1x1 = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=1, padding=0, bias=False)
        self.bn2 = norm_layer(self.reduced_channels)

        self.inter_sma = InterSMA(u, t, self.reduced_channels, norm_layer)

        self.relu = nn.ReLU(inplace=True)

        self.conv_u = nn.Conv2d(ch_in, ch_in, kernel_size=(3, 1), padding=(1, 0), bias=False)

    def forward(self, x):
        # but, c, h, w = x.shape
        # bu,t,c,h,w
        x_ = x.reshape((-1, self.t) + x.shape[1:])
        bu, t, c, h, w = x_.shape
        x_ = x_.reshape(-1, self.u, t, c, h, w)
        # b, u, t, c, h, w = x_.shape
        # todo b,t,c,u,h,w 模块输入shape, 用于点乘
        feat_raw = x_.permute(0, 2, 3, 1, 4, 5).contiguous()
        # b,t,c,u,1,1
        feat = self.avg_pool(feat_raw)
        # b,t,c,u
        feat = feat.squeeze().squeeze()
        # b,c,u,t
        feat = feat.permute(0, 2, 3, 1).contiguous()

        # 分支1
        # x1 = feat.squeeze().squeeze()
        # b,c,u,t
        x1 = self.bn1(self.conv3x1(feat))  # 2, 32,2,4
        # 没有bn层
        x1 = torch.sigmoid(self.conv1x1(x1))

        # 分支2
        # b, t, c, u, 1, 1
        # x2 = feat.squeeze(-1)
        # b,u,c,t,1
        # x2 = x2.permute(0, 3, 2, 1, 4).contiguous()
        # bu,c,t,1
        # x2 = x2.reshape(-1, c, t, 1)

        # b,u,c,t
        feat_2 = feat.permute(0, 2, 1, 3).contiguous()
        # b,u,c,t,1 -> bu,c,t,1
        b, u, c, t = feat_2.shape
        feat_2 = feat_2.squeeze(-1).reshape(-1, c, t, 1)
        # b,c,1,t
        x2 = self.inter_sma(feat_2)
        # b,c,u,t
        x12 = x1 * x2
        # b,t,c,u
        x12 = x12.permute(0, 3, 1, 2).contiguous()
        # b,t,c,u,1,1
        x12 = x12.unsqueeze(-1).unsqueeze(-1)
        # x2_avg = self.avg_pool(x1 * x2)
        # b,t,c,u,h,w
        x_merge = x12 * feat_raw + feat_raw

        x_return = x_merge.permute(0, 3, 1, 2, 4, 5).contiguous().reshape(-1, c, h, w)
        x_return = self.relu(self.bn2(self.conv_u(x_return)))

        return x_return


# class IntraSIBlock(nn.Module):
#     def __init__(self):
#         super(IntraSIBlock, self).__init__()
#         ch_in = 1
#         self.conv_1 = nn.Conv2d(ch_in, ch_in, kernel_size=1, padding=0, bias=False)
#         self.bn_1 = nn.BatchNorm2d(ch_in)
#         self.conv_2 = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, bias=False)
#         self.bn_2 = nn.BatchNorm2d(ch_in)
#         self.conv_3 = nn.Conv2d(ch_in, ch_in, kernel_size=1, padding=0, bias=False)
#         self.bn_3 = nn.BatchNorm2d(ch_in)
#
#         self.intra_sim = IntraSIM()
#         self.inter_sim = InterSIM()
#
#     def forward(self, input):
#         x = self.conv_1(input)
#         x = self.intra_sim(x)
#         x = self.bn_2(self.conv_2(x))
#         x = self.bn_3(self.conv_3(x))
#
#         x = x + input
#
#         return x
#
#
# class InterSIBlock(nn.Module):
#     def __init__(self):
#         super(InterSIBlock, self).__init__()
#         ch_in = 1
#         self.conv_1 = nn.Conv2d(ch_in, ch_in, kernel_size=1, padding=0, bias=False)
#         self.bn_1 = nn.BatchNorm2d(ch_in)
#         self.conv_2 = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, bias=False)
#         self.bn_2 = nn.BatchNorm2d(ch_in)
#         self.conv_3 = nn.Conv2d(ch_in, ch_in, kernel_size=1, padding=0, bias=False)
#         self.bn_3 = nn.BatchNorm2d(ch_in)
#
#         self.inter_sim = InterSIM()
#
#     def forward(self, input):
#         x = self.conv_1(input)
#         x = self.inter_sim(x)
#         x = self.bn_2(self.conv_2(x))
#         x = self.bn_3(self.conv_3(x))
#
#         x = x + input
#
#         return x


class Bottleneck(nn.Module):
    """
    SCNet SCBottleneck. Variant for ResNet Bottlenect.
    """
    expansion = 4
    pooling_r = 4  # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, u, t, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, bottleneck_width=32,
                 avd=False, dilation=1, is_first=False,
                 norm_layer=None):
        """

        Args:
            u: 每个视频分段数量
            t: 每个分段中图片数量
            inplanes:
            planes:
            stride:
            downsample:
            cardinality:
            bottleneck_width:
            avd:
            dilation:
            is_first:
            norm_layer:
        """
        super(Bottleneck, self).__init__()
        # group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        # self.conv1_a = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        # self.bn1_a = norm_layer(group_width)
        # self.conv1_b = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        # self.bn1_b = norm_layer(group_width)
        # self.avd = avd and (stride > 1 or is_first)
        self.stide = stride

        # todo 似乎只有inplanes与group_width值一样时,才能执行残差和
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality

        self.conv1_a1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn_a1 = norm_layer(group_width)
        self.intra_sim1 = IntraSIM(u, t, group_width, norm_layer)
        # 使用卷积进行下采样
        self.conv3_a2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=dilation, bias=False)
        self.bn_a2 = norm_layer(group_width)
        self.conv1_a3 = nn.Conv2d(group_width, inplanes, kernel_size=1, bias=False)
        self.bn_a3 = norm_layer(inplanes)

        self.conv1_a4 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn_a4 = norm_layer(group_width)
        self.inter_sim1 = InterSIM(u, t, group_width, norm_layer)
        self.conv3_a5 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_a5 = norm_layer(group_width)
        self.conv1_a6 = nn.Conv2d(group_width, inplanes, kernel_size=1, bias=False)
        self.bn_a6 = norm_layer(inplanes)

        self.conv1_b1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn_b1 = norm_layer(group_width)
        self.intra_sim2 = IntraSIM(u, t, group_width, norm_layer)
        self.conv3_b2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b2 = norm_layer(group_width)
        self.conv1_b3 = nn.Conv2d(group_width, inplanes, kernel_size=1, bias=False)
        self.bn_b3 = norm_layer(inplanes)

        self.conv1_b4 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn_b4 = norm_layer(group_width)
        self.inter_sim2 = InterSIM(u, t, group_width, norm_layer)
        self.conv3_b5 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b5 = norm_layer(group_width)
        self.conv1_b6 = nn.Conv2d(group_width, inplanes, kernel_size=1, bias=False)
        self.bn_b6 = norm_layer(inplanes)

        # self.tim = TIMModule(group_width, n_segment=num_segments)
        # self.shift = ShiftModule(group_width, n_segment=num_segments, n_div=8, mode='shift')
        self.inplanes = inplanes
        self.planes = planes
        # self.ism = ISMModule()
        # self.shift = ShiftModule(group_width, n_segment=num_segments, n_div=8, mode='shift')

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation

        # 对步长不为1的残差层进行下采样
        self.avd_layer = nn.AvgPool2d(3, stride, padding=1)

    def forward(self, x):
        """Forward func which splits the input into two branchs a and b.
        a: trace features
        b: spatial features
        """
        residual1 = x
        x = self.relu(self.bn_a1(self.conv1_a1(x)))
        x = self.intra_sim1(x)
        x = self.relu(self.bn_a2(self.conv3_a2(x)))
        x = self.relu(self.bn_a3(self.conv1_a3(x)))

        if self.stide > 1:
            residual1 = self.avd_layer(residual1)
        x = residual1 + x

        residual2 = x
        x = self.relu(self.bn_a4(self.conv1_a4(x)))
        x = self.inter_sim1(x)
        x = self.relu(self.bn_a5(self.conv3_a5(x)))
        x = self.relu(self.bn_a6(self.conv1_a6(x)))
        x = residual2 + x

        residual3 = x
        x = self.relu(self.bn_b1(self.conv1_b1(x)))
        x = self.intra_sim2(x)
        x = self.relu(self.bn_b2(self.conv3_b2(x)))
        x = self.relu(self.bn_b3(self.conv1_b3(x)))
        x = residual3 + x

        residual4 = x
        x = self.relu(self.bn_b4(self.conv1_b4(x)))
        x = self.inter_sim2(x)
        x = self.relu(self.bn_b5(self.conv3_b5(x)))
        x = self.relu(self.bn_b6(self.conv1_b6(x)))
        out = residual4 + x

        if self.downsample is not None:
            out = self.downsample(out)

        return out


class SCINet(nn.Module):
    def __init__(self, u, t, block, layers, groups=1, bottleneck_width=32,
                 num_classes=1000, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 avd=False, norm_layer=nn.BatchNorm2d):
        """SCNet, a variant based on ResNet.

        Args:
            num_segments (int):
                Number of input frames.
            block (class):
                Class for the residual block.
            layers (list):
                Number of layers in each block.
            num_classes (int, optional):
                Number of classification class.. Defaults to 1000.
            dilated (bool, optional):
                Whether to apply dilation conv. Defaults to False.
            dilation (int, optional):
                The dilation parameter in dilation conv. Defaults to 1.
            deep_stem (bool, optional):
                Whether to replace 7x7 conv in input stem with 3 3x3 conv. Defaults to False.
            stem_width (int, optional):
                Stem width in conv1 stem. Defaults to 64.
            avg_down (bool, optional):
                Whether to use AvgPool instead of stride conv when downsampling in the bottleneck. Defaults to False.
            avd (bool, optional):
                The avd parameter for the block Defaults to False.
            norm_layer (class, optional):
                Normalization layer. Defaults to nn.BatchNorm2d.
        """
        super(SCINet, self).__init__()

        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.avd = avd
        # self.u, self.t = num_segments
        self.u = u
        self.t = t

        conv_layer = nn.Conv2d
        if deep_stem:
            # 使用3x3卷积替代7x7
            self.conv1 = nn.Sequential(
                    conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                    norm_layer(stem_width),
                    nn.ReLU(inplace=True),
                    conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                    norm_layer(stem_width),
                    nn.ReLU(inplace=True),
                    conv_layer(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3,
                                    bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # todo 使用卷积代替maxpool2d
        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.inplanes)

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        # 不使用空洞卷积
        if dilated or dilation == 4:
            #     self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            #                                    dilation=2, norm_layer=norm_layer)
            #     self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            #                                    dilation=4, norm_layer=norm_layer)
            # elif dilation == 2:
            #     self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
            #                                    dilation=1, norm_layer=norm_layer)
            #     self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            #                                    dilation=2, norm_layer=norm_layer)
            pass
        else:
            # 走下面两个layer
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        # self.fc2 = nn.Linear(num_classes * 64, num_classes)

        # 模型初始化方式
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    is_first=True):
        """
        Core function to build layers.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            # todo stil中使用队列avg_down, 新代码中不想使用这个avgpool2d
            if self.avg_down:
                # if dilation == 1:
                #     down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                #                                     ceil_mode=True, count_include_pad=False))
                # else:
                #     down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                #                                     ceil_mode=True, count_include_pad=False))
                # down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                #                              kernel_size=1, stride=1, bias=False))
                pass
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        # dilation==1时是正常卷积,当前代码中不使用空洞卷积
        if dilation == 1 or dilation == 2:
            layers.append(block(self.u, self.t, self.inplanes, planes, stride, downsample=downsample,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=1, is_first=is_first,
                                norm_layer=norm_layer))
        # elif dilation == 4:
        #     layers.append(block(self.u, self.t, self.inplanes, planes, stride, downsample=downsample,
        #                         cardinality=self.cardinality,
        #                         bottleneck_width=self.bottleneck_width,
        #                         avd=self.avd, dilation=2, is_first=is_first,
        #                         norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.u, self.t, self.inplanes, planes,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        # 使用卷积替代最大池化进行下采样
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.relu(self.bn2(self.conv2(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def logits(self, features):
        x = self.avgpool(features)
        x = x.reshape(x.shape[0], -1)
        # todo 增加一个全连接层
        x = self.fc1(x)
        # x = self.fc2(x)

        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def scnet50_v1d(u, t, pretrained=False, **kwargs):
    """
    SCNet backbone, which is based on ResNet-50
    Args:
        num_segments (int):
            Number of input frames.
        pretrained (bool, optional):
            Whether to load pretrained weights.
    """
    model = SCINet(u, t, Bottleneck, [3, 4, 6, 3],
                   deep_stem=True, stem_width=32, avg_down=False,
                   avd=True, **kwargs)
    if pretrained:
        # todo 修改模型读入方式
        model.load_state_dict(model_zoo.load_url(model_urls['scnet50_v1d']), strict=False)

    return model

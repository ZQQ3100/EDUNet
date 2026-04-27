import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import sqrt
import numpy as np
import numpy.fft as fft
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from typing import Tuple
from typing import Dict
import time

def conv_layer(inDim, outDim, ks, s, p, norm_layer='none'):
    ## convolutional layer
    conv = nn.Conv2d(inDim, outDim, kernel_size=ks, stride=s, padding=p)
    relu = nn.ReLU(True)
    assert norm_layer in ('batch', 'instance', 'none')
    if norm_layer == 'none':
        seq = nn.Sequential(*[conv, relu])
    else:
        if (norm_layer == 'instance'):
            norm = nn.InstanceNorm2d(outDim, affine=False, track_running_stats=False) # instance norm
        else:
            momentum = 0.1
            norm = nn.BatchNorm2d(outDim, momentum = momentum, affine=True, track_running_stats=True)
        seq = nn.Sequential(*[conv, norm, relu])
    return seq

def LDI_subNet(inDim=32, outDim=32, norm='none'):
    ## LDI network
    convBlock1 = conv_layer(inDim,64,3,1,1)
    convBlock2 = conv_layer(64,128,3,1,1,norm)
    convBlock3 = conv_layer(128,64,3,1,1,norm)
    convBlock4 = conv_layer(64,32,3,1,1,norm)
    conv = nn.Conv2d(32, outDim, 3, 1, 1)
    seq = nn.Sequential(*[convBlock1, convBlock2, convBlock3, convBlock4, conv])
    return seq

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ## LDI network
        self.LDI1 = LDI_subNet(32, 32)
        self.LDI2 = LDI_subNet(32, 32)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

        self.KT = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1))
        self.K = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                               nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                               nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                               nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1))

        self.image_d = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.endconv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
       
        self.eta_R = nn.Parameter(torch.tensor(0.01), requires_grad=True)
        self.eta_I = nn.Parameter(torch.tensor(0.01), requires_grad=True)
        self.lambda1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.lambda2 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.beta = 0.005
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)
        self.theta = nn.Parameter(torch.ones(1), requires_grad=True)

        torch.nn.init.normal_(self.eta_R, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.eta_I, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.lambda1, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.lambda2, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.gamma, mean=0.5, std=0.01)
        torch.nn.init.normal_(self.theta, mean=0.5, std=0.01)

        self._init_weights()

    def _init_weights(self):
        # 第一步：通用初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

        # 第二步：对称性初始化
        with torch.no_grad():
            self.KT[0].weight.data = self.K[6].weight.permute(1, 0, 2, 3).clone()
            self.KT[2].weight.data = self.K[4].weight.permute(1, 0, 2, 3).clone()
            self.KT[4].weight.data = self.K[2].weight.permute(1, 0, 2, 3).clone()
            self.KT[6].weight.data = self.K[0].weight.permute(1, 0, 2, 3).clone()

    def soft_threshold(self, g, beta):
        """软阈值函数"""
        return torch.sign(g) * self.relu(torch.abs(g) - beta)


    def gradient_operator(self, I):
        """
        多通道图像梯度计算 (前向差分)
        输入: img [B, C, H, W]
        输出: grad [B, 2*C, H, W] (每通道的x和y梯度拼接)
        """
        B, C, H, W = I.shape
        device = I.device
        kernel_x = torch.tensor([[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]], dtype=torch.float32, device=device)
        kernel_y = torch.tensor([[[0, -1, 0], [0, 1, 0], [0, 0, 0]]], dtype=torch.float32, device=device)
        kernel_x = kernel_x.repeat(C, 1, 1, 1)
        kernel_y = kernel_y.repeat(C, 1, 1, 1)
        grad_x = F.conv2d(I, kernel_x, padding=1, groups=C)
        grad_y = F.conv2d(I, kernel_y, padding=1, groups=C)
        grad = torch.stack([grad_x, grad_y], dim=2)
        grad = grad.view(B, 2 * C, H, W)
        return grad

    def divergence_operator(self, grad):
        """
        多通道散度计算 (后向差分)
        输入: grad [B, 2*C, H, W] (必须是由gradient_operator输出的格式)
        输出: div [B, C, H, W]
        """
        B, C2, H, W = grad.shape
        C = C2 // 2
        device = grad.device
        kernel_x = torch.tensor([[[0, 0, 0], [1, -1, 0], [0, 0, 0]]], dtype=torch.float32, device=device)
        kernel_y = torch.tensor([[[0, 1, 0], [0, -1, 0], [0, 0, 0]]], dtype=torch.float32, device=device)
        kernel_x = kernel_x.repeat(C, 1, 1, 1)
        kernel_y = kernel_y.repeat(C, 1, 1, 1)
        grad_x = grad[:, 0::2]
        grad_y = grad[:, 1::2]
        div_x = F.conv2d(grad_x, kernel_x, padding=1, groups=C)
        div_y = F.conv2d(grad_y, kernel_y, padding=1, groups=C)
        return div_x + div_y


    def updata_R(self, R, E, B, I):
        if B.shape != R.shape:
            print(B.shape)
            print(R.shape)
            raise ValueError("B should have the same shape as R and E after element-wise multiplication.")
        eta_r = F.softplus(self.eta_R)
        part1 = self.lambda2 * ((R * E - B) * E)
        part2 = self.gamma * (R - I)
        R_next = R - 2 * eta_r * (part1 + part2)
        return R_next


    def updata_g(self, I):
        grad_I = self.gradient_operator(I)
        threshold = self.beta / self.theta
        g = self.soft_threshold(grad_I, threshold)
        return g

    def updata_I(self, I, B, R, g):
        eta_i = F.softplus(self.eta_I)
        KI = self.K(I)
        Res1 = self.KT(KI-B)
        grad_I = self.gradient_operator(I)
        I_next = I - 2 * eta_i * (
                self.lambda1 * Res1 +
                self.gamma * (I - R) -
                self.theta * self.divergence_operator(grad_I - g)
        )
        return I_next

    def forward(self, B, eventstream):
        device = next(self.parameters()).device
        I = self.image_d(B)
        event_1 = eventstream[:, range(0, 32), :, :]
        E_1 = self.LDI1(event_1)
        event_2 = eventstream[:, range(32, 64), :, :]
        E_2 = self.LDI2(event_2)

        E = 0.5 * E_1 + 0.5 * E_2
        E = self.relu(E) + self.sigmoid(E)
        R = self.image_d(B).detach().clone()
        g = torch.zeros_like(self.gradient_operator(I), device=device)
        if B.shape != R.shape:
            B = B.expand_as(R)


        for i in range(10):
            R = self.updata_R(R, E, B, I)
            g = self.updata_g(I)
            I = self.updata_I(I, B, R, g)
        out = self.endconv(I)
        L = B / (E + 1e-8)
        L = self.endconv(L)
        loss_consistency = F.mse_loss(self.K(self.KT(B)), B) + F.mse_loss(self.KT(self.K(B)), B)
        return out, L, loss_consistency
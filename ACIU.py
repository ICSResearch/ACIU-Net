
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch



class GFA(nn.Module):

    def __init__(self):
        super().__init__()
        self.f_saf = nn.AdaptiveAvgPool2d(1)
        self.gamma = nn.Parameter(torch.zeros(1, 2, 1, 1))
        self.xi = nn.Parameter(torch.zeros(1, 2, 1, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * 2, -1, h, w)
        f = x * self.f_saf(x)
        p = f.sum(dim=1, keepdim=True)
        p = p.view(b * 2, -1)
        p = p - p.mean(dim=1, keepdim=True)
        std = p.std(dim=1, keepdim=True) + 1e-5
        p = p / std
        p = p.view(b, 2, h, w)
        p = p * self.gamma + self.xi
        p = p.view(b * 2, 1, h, w)
        x = x * self.sig(p)
        x = x.view(b, c, h, w)

        return x



class FFT_Mask_ForBack(torch.nn.Module):
    def __init__(self):
        super(FFT_Mask_ForBack, self).__init__()

    def forward(self, x, full_mask):
        full_mask = full_mask[..., 0]
        x_in_k_space = torch.fft.fft2(x)
        masked_x_in_k_space = x_in_k_space * full_mask.view(1, 1, *(full_mask.shape))
        masked_x = torch.real(torch.fft.ifft2(masked_x_in_k_space))
        return masked_x




class MFE(torch.nn.Module):
    def __init__(self):
        super(MFE, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.gfa = GFA()

        self.blk1 = ResBlk(96, 32)
        self.blk2 = ResBlk(32, 64)
        self.blk3 = ResBlk(64, 32)
        self.blk4 = ResBlk(32, 32)
        self.inc_f = Inception()


    def forward(self, x, fft_forback, PhiTb, mask):
        x = x - self.lambda_step * fft_forback(x, mask)
        x = x + self.lambda_step * PhiTb
        x_input = x
        x_D = self.inc_f(x_input)
        x_D = F.relu(x_D)
        x = self.blk1(x_D)
        x = self.blk2(x)
        x = self.gfa(x)
        x = torch.mul(torch.sign(x), F.relu(torch.abs(x) - self.soft_thr))
        x = self.gfa(x)
        x = self.blk3(x)
        x_G = self.blk4(x)

        x_pred = x_input + x_G


        return x_pred

class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.branch3 = nn.Sequential(

            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        outputs = [branch1, branch2, branch3]
        return torch.cat(outputs, dim=1)

class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out):

        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3,  padding=1)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)

        # self.extra = nn.Sequential()
        # if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
        self.extra = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0),

        )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.extra(x) + out
        return out




class ACIUNet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ACIUNet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.fft_forback = FFT_Mask_ForBack()

        for i in range(LayerNo):
            onelayer.append(MFE())

        self.fcs = nn.ModuleList(onelayer)
        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, PhiTb, mask):


        PhiTb = F.conv2d(PhiTb, self.conv_D, padding=1)
        x = PhiTb


        for i in range(self.LayerNo):
            x = self.fcs[i](x, self.fft_forback, PhiTb, mask)

        x_final = F.conv2d(x, self.conv_G, padding=1)
        return x_final



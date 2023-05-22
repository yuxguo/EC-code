import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from .modules import *
from torchvision.models.resnet import Bottleneck, BasicBlock
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor



class SymbolModel(nn.Module):
    # input: [B, 8 + 8, 4, 10]
    # output: image embedding[B, 8 + 8, image_embedding] (image_embedding = 512)
    def __init__(self, args):
        super().__init__()
        self.mlp = MLPModel(args.symbol_onehot_dim, args.image_embedding_dim // args.symbol_attr_dim, args.symbol_model_hidden_dims)
        self.fc_res_block = FCResBlock(args.image_embedding_dim // args.symbol_attr_dim)
        self.d4_symbol = False
        if self.d4_symbol:
            self.convert_ = nn.Sequential(*[nn.ReLU(), nn.Linear(320, 80)]) # FIXME

    def forward(self, symbol):
        # symbol [B, c, symbol_attr_dim, symbol_onehot_dim]
        # print(symbol.size())
        B, C, = symbol.size()[:2]
        output = self.mlp(symbol)
        output = self.fc_res_block(output)
        output = output.view(B, C, -1)
        if self.d4_symbol:
            output = self.convert_(output)
        return output

class VisualModel(nn.Module):
    # input: RAW image[B, 8 + 8, raw_size, raw_size]
    # output: image embedding[B, 8 + 8, image_embedding] (image_embedding = 512)
    def __init__(self, args):
        super().__init__()
        self.use_resnet = args.use_resnet
        if self.use_resnet:
            self.cnn = Resnet18()
        else:
            self.cnn = ConvNet()
            # self.cnn = RPMResNet()
            self.spatial_convert_mlp = nn.Linear(100, 80) # FIXME: Don't use magic number
            self.shared_group_mlp = SharedGroupMLP(**args.visual_shared_mlp_configs)
        
        self.convert = False
        if self.use_resnet and args.image_embedding_dim != 512:
            self.convert = True
            self.convert_mlp = nn.Linear(512, args.image_embedding_dim)
        elif not self.use_resnet and args.image_embedding_dim != 80:
            self.convert = True
            self.convert_mlp = nn.Linear(80, args.image_embedding_dim)
        
        self.fc_block = FCResBlock(args.image_embedding_dim) 

    def forward(self, images):
        b, n, h, w = images.size()
        images = images.view(b * n, 1, h, w)
        if self.use_resnet:
            output = self.cnn(images)
        else:
            output = self.cnn(images) # (b * n, 32, 10, 10)
            output = output.flatten(-2, -1) # (b * n, 32, 100)
            output = self.spatial_convert_mlp(output) # (b * n, 32, 80)
            # output = output.permute(0, 2, 1).contiguous()
            # print(output.shape)
            output = self.shared_group_mlp(output) # (b * n, 256)

        output = output.view(b, n, -1)
        if self.convert:
            output = self.convert_mlp(output)
        output = torch.relu(output)
        self.fc_block(output)
        return output


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ConvNet(nn.Module):
    def __init__(self,
            input_dim=1,
            hidden_dims=[16, 16, 32, 32],
            repeats=None,
            kernels=3,
            residual_link=True,
            image_size=[160, 160],
            flatten=False,
            use_layer_norm=False):
        super().__init__()
        h, w = image_size

        if type(kernels) is list:
            if len(kernels) == 1:
                kernel_size = kernels[0]
            else:
                kernel_size = tuple(kernels)
        else:
            kernel_size = kernels

        if repeats is None:
            repeats = [1 for i in range(len(hidden_dims))]
        else:
            assert len(repeats) == len(hidden_dims)

        conv_blocks = []
        current_dim = input_dim
        # NOTE: The last hidden dim is the output dim
        for rep, hidden_dim in zip(repeats, hidden_dims):
            block = ConvBlock(current_dim, hidden_dim, h, w,
                repeats=rep,
                kernel_size=kernel_size,
                residual_link=residual_link,
                use_layer_norm=use_layer_norm)
            current_dim = hidden_dim
            conv_blocks.append(block)
            h, w = block.output_size

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.flatten = flatten
        self.output_dim = hidden_dims[-1]
        self.output_image_size = (h, w)
        # self.output_size = hidden_dims[-1] * h * w

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        # default: image_size = (80, 80)
        # batch, input_dim, 80, 80
        # batch, hidden_dim[0], 40, 40
        # batch, hidden_dim[1], 20, 20
        # batch, hidden_dim[2], 10, 10
        # batch, hidden_dim[3], 5, 5
        if self.flatten:
            x = x.flatten(1, -1)
            # batch, hidden_dim[4] * 5 * 5
        return x


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, h, w, repeats=1,
            kernel_size=3, padding=1, residual_link=True, use_layer_norm=False):
        super().__init__()
        convs = []
        norms = []
        if type(kernel_size) is int:
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = kernel_size

        current_dim = input_dim
        for i in range(repeats):
            stride = 1
            if i == 0:
                # The reduction conv
                stride = 2
                h = (h + 2 * padding - kh + stride) // stride
                w = (w + 2 * padding - kw + stride) // stride
            convs.append(nn.Conv2d(current_dim, output_dim,
                kernel_size=kernel_size, stride=stride, padding=padding))
            current_dim = output_dim
            if use_layer_norm:
                norms.append(nn.LayerNorm([current_dim, h, w]))
            else:
                norms.append(nn.BatchNorm2d(current_dim))

        self.residual_link = residual_link
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.output_size = (h, w)

    def forward(self, x):
        is_reduction = True
        for conv, norm in zip(self.convs, self.norms):
            # ConvNormReLU
            _ = x
            _ = conv(_)
            _ = norm(_)
            _ = F.relu(_)
            if is_reduction or not self.residual_link:
                x = _
            else:
                x = x + _
            is_reduction = False

        return x


class IdentityBlock(nn.Module):
    def __init__(self):
        super(IdentityBlock, self).__init__()

    def forward(self, x):
        return x


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.cnn = torchvision.models.resnet18(pretrained=False)
        self.cnn.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.fc = IdentityBlock()

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * c, 1, h, w)
        x = self.cnn(x)  # [B, 16, 512]
        x = x.view(b, c, -1)
        return x


class IBlock(nn.Module):
    def __init__(self, ic, oc, ks, stride, padding):
        super().__init__()
        seq = [nn.Conv2d(ic, oc, ks, stride, padding), nn.BatchNorm2d(oc), nn.ReLU(inplace=True), nn.Conv2d(oc, oc, ks, stride, padding), nn.BatchNorm2d(oc)]
        self.conv = nn.Sequential(*seq)

    def forward(self, x):
        return x + self.conv(x)

class DBlock(nn.Module):
    def __init__(self, ic, oc, ks, stride, padding):
        super().__init__()
        seq = [nn.Conv2d(ic, oc, ks, stride[0], padding), nn.BatchNorm2d(oc), nn.ReLU(inplace=True), nn.Conv2d(oc, oc, ks, stride[1], padding), nn.BatchNorm2d(oc)]
        self.conv = nn.Sequential(*seq)
        sct = [nn.Conv2d(ic, oc, 1, stride[0], 0), nn.BatchNorm2d(oc)]
        self.shortcut = nn.Sequential(*sct)

    def forward(self, x):
        return self.shortcut(x) + self.conv(x)


class RPMResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 8
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = IBlock(self.inplanes, self.inplanes, 3, 1, 1)
        self.layer2 = DBlock(8, 16, 3, [2, 1], 1)
        self.layer3 = DBlock(16, 32, 3, [2, 1], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


if __name__ == "__main__":
    model = RPMResNet().cuda(1)
    x = torch.rand(32, 1, 160, 160).cuda(1)

    print(model(x).size())
import torch
import torch.nn as nn
#from .utils import load_state_dict_from_url
import torchvision
import numpy as np
from torchvision import transforms

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']



def bilinear_init(in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=35, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        
        
        self.res18_model = torchvision.models.resnet18(pretrained=True)
        self.res18_conv = nn.Sequential(*list(self.res18_model.children())[:-2])
        

        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.score_fr = nn.Conv2d(512, num_classes, 1)
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32,
                                            bias=False)
        
    #     if norm_layer is None:
    #         norm_layer = nn.BatchNorm2d
    #     self._norm_layer = norm_layer

    #     self.inplanes = 64
    #     self.dilation = 1
    #     if replace_stride_with_dilation is None:
    #         # each element in the tuple indicates if we should replace
    #         # the 2x2 stride with a dilated convolution instead
    #         replace_stride_with_dilation = [False, False, False]
    #     if len(replace_stride_with_dilation) != 3:
    #         raise ValueError("replace_stride_with_dilation should be None "
    #                          "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
    #     self.groups = groups
    #     self.base_width = width_per_group
    #     self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
    #                            bias=False)
    #     self.bn1 = norm_layer(self.inplanes)
    #     self.relu = nn.ReLU(inplace=True)
    #     self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    #     self.layer1 = self._make_layer(block, 64, layers[0])
    #     self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
    #                                    dilate=replace_stride_with_dilation[0])
    #     self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
    #                                    dilate=replace_stride_with_dilation[1])
    #     self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
    #                                    dilate=replace_stride_with_dilation[2])
    #     #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    #     self.avgpool = nn.AvgPool2d(7, stride=1)
    #     #self.fc = nn.Linear(512 * block.expansion, num_classes)

    #     self.score_fr = nn.Conv2d(512, num_classes, 1)
    #     self.upscore = nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32,
    #                                       bias=False)


    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    #     # Zero-initialize the last BN in each residual branch,
    #     # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    #     # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    #     if zero_init_residual:
    #         for m in self.modules():
    #             if isinstance(m, Bottleneck):
    #                 nn.init.constant_(m.bn3.weight, 0)
    #             elif isinstance(m, BasicBlock):
    #                 nn.init.constant_(m.bn2.weight, 0)

    # def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    #     norm_layer = self._norm_layer
    #     downsample = None
    #     previous_dilation = self.dilation
    #     if dilate:
    #         self.dilation *= stride
    #         stride = 1
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             conv1x1(self.inplanes, planes * block.expansion, stride),
    #             norm_layer(planes * block.expansion),
    #         )

    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
    #                         self.base_width, previous_dilation, norm_layer))
    #     self.inplanes = planes * block.expansion
    #     for _ in range(1, blocks):
    #         layers.append(block(self.inplanes, planes, groups=self.groups,
    #                             base_width=self.base_width, dilation=self.dilation,
    #                             norm_layer=norm_layer))

    #     return nn.Sequential(*layers)

    # def _forward_impl(self, x):
    #     # See note [TorchScript super()]
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)

    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)

    #     x = self.avgpool(x)
    #     #x = torch.flatten(x, 1)
    #     #x = self.fc(x)
    #     x = self.score_fr(x)
    #     x = self.upscore(x)

    #     return x

    # def forward(self, x):
    #     return self._forward_impl(x)


    def forward(self, x):  
        x = self.res18_conv(x)
        x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)
        x = self.score_fr(x)
        x = self.upscore(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
 
 
 
class ResNet18(nn.Module):
    
    def __init__(self, model, num_classes=35):
        super(ResNet18, self).__init__()
        self.pad = nn.ZeroPad2d(100)
        self.res18_model = model
        self.res18_conv = nn.Sequential(*list(self.res18_model.children())[:-2])
    
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.score_fr = nn.Conv2d(512, num_classes, 1)
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32,
                                            bias=False)
    def forward(self, x):  
        img_size = x.size()[2], x.size()[3]
        print("img_size :", img_size)
        x = self.pad(x)
        print("###################")
        print(x.shape)
        x = self.res18_conv(x)
        x = self.avgpool(x)
        print("is it 14*14*35 : ", x.shape)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)
        x = self.score_fr(x)
       
        x = self.upscore(x)
        
        #x = x[:, :, 100:100 + x.size()[2], 100:100 + x.size()[3]]
        #x = x[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()
        transform = torchvision.transforms.CenterCrop(img_size)
        x = transform(x)
        print("output img_size :", x.shape)
        return x

    
def FCN32(backbone):
    return ResNet18()




class FCN16(nn.Module):
    
    def __init__(self, backbone, in_channel=512, num_classes=35):
        super(FCN16, self).__init__()
        self.backbone = backbone
        self.cls_num = num_classes

        self.relu    = nn.ReLU(inplace=True)
        self.Conv1x1 = nn.Conv2d(in_channel, self.cls_num, kernel_size=1)
        self.Conv1x1_x4 = nn.Conv2d(int(in_channel/2), self.cls_num, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.cls_num)
        self.DCN2 = nn.ConvTranspose2d(self.cls_num, self.cls_num, kernel_size=4, stride=2, dilation=1, padding=1)
        self.DCN2.weight.data = bilinear_init(self.cls_num, self.cls_num, 4)
        self.dbn2 = nn.BatchNorm2d(self.cls_num)

        self.DCN16 = nn.ConvTranspose2d(self.cls_num, self.cls_num, kernel_size=32, stride=16, dilation=1, padding=8)
        self.DCN16.weight.data = bilinear_init(self.cls_num, self.cls_num, 32)
        self.dbn16 = nn.BatchNorm2d(self.cls_num)
    

    def forward(self, x):
        x0, x1, x2, x3, x4, x5, x6 = self.backbone(x)
        x = self.bn1(self.relu(self.Conv1x1(x5)))
        x4 = self.bn1(self.relu(self.Conv1x1_x4(x4)))
        x = self.dbn2(self.relu(self.DCN2(x)))
        x = x + x4
        x = self.dbn16(self.relu(self.DCN16(x)))
        
        return x 
import math
import torch.nn as nn
import torch
import torchvision
from torch.cuda.amp import autocast
from torchvision.models.feature_extraction import create_feature_extractor

class ConvNextOri(nn.Module):
    def __init__(self, label_dim=309, pretrain=True, model_id=0, audioset_pretrain=False):
        super().__init__()
        print('now train a convnext model ' + str(model_id))
        model_id = int(model_id)
        if model_id == 0:
            self.model = torchvision.models.convnext_tiny(pretrained=pretrain)
        elif model_id == 1:
            self.model = torchvision.models.convnext_small(pretrained=pretrain)
        elif model_id == 2:
            self.model = torchvision.models.convnext_base(pretrained=pretrain)
        elif model_id == 3:
            self.model = torchvision.models.convnext_large(pretrained=pretrain)
        hid_dim = [768, 768, 1024, 1536]
        self.model = torch.nn.Sequential(*list(self.model.children()))
        self.model[-1][-1] = torch.nn.Linear(hid_dim[model_id], label_dim)

        new_proj = torch.nn.Conv2d(1, 192, kernel_size=(4, 4), stride=(4, 4), bias=True)
        print('conv1 get from pretrained model.')
        new_proj.weight = torch.nn.Parameter(torch.sum(self.model[0][0][0].weight, dim=1).unsqueeze(1))
        new_proj.bias = self.model[0][0][0].bias
        self.model[0][0][0] = new_proj

    def forward(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        out = self.model(x)
        return out

    def feature_extract(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.model[:-2](x)
        x = torch.mean(x, dim=2)
        x = x.transpose(1, 2)
        return x

class EffNetOri(nn.Module):
    def __init__(self, label_dim=527, pretrain=True, model_id=0):
        super().__init__()
        b = int(model_id)
        print('now train a effnet-b{:d} model'.format(b))
        if b == 7:
            self.model = torchvision.models.efficientnet_b7(pretrained=pretrain)
        elif b == 6:
            self.model = torchvision.models.efficientnet_b6(pretrained=pretrain)
        elif b == 5:
            self.model = torchvision.models.efficientnet_b5(pretrained=pretrain)
        elif b == 4:
            self.model = torchvision.models.efficientnet_b4(pretrained=pretrain)
        elif b == 3:
            self.model = torchvision.models.efficientnet_b3(pretrained=pretrain)
        elif b == 2:
            self.model = torchvision.models.efficientnet_b2(pretrained=pretrain)
        elif b == 1:
            self.model = torchvision.models.efficientnet_b1(pretrained=pretrain)
        elif b == 0:
            self.model = torchvision.models.efficientnet_b0(pretrained=pretrain)

        new_proj = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        print('conv1 get from pretrained model.')
        new_proj.weight = torch.nn.Parameter(torch.sum(self.model.features[0][0].weight, dim=1).unsqueeze(1))
        new_proj.bias = self.model.features[0][0].bias
        self.model.features[0][0] = new_proj
        self.model = create_feature_extractor(self.model, {'features.8': 'mout'})
        hid_dim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        self.cla = torch.nn.Sequential(nn.LayerNorm(hid_dim[int(model_id)]), nn.Linear(hid_dim[int(model_id)], label_dim))

    def forward(self, x, valid_len):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.model(x)['mout']
        for i in range(x.shape[0]):
            x[i, :, :, 0] = torch.mean(x[i, :, :, :math.ceil(valid_len[i] / 33)], dim=-1)
        x = x[:, :, :, 0]
        x = torch.mean(x, dim=[2])
        out = self.cla(x)
        return out

    def forward_old(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.model(x)['mout']
        x = torch.mean(x, dim=[2, 3])
        out = self.cla(x)
        return out

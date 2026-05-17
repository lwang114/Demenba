# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import whisper
from torchvision.models.feature_extraction import create_feature_extractor


class WhisperReader(object):
    def __init__(self, model_name, download_root, layers):
        self.model = whisper.load_model(
            model_name.split('_')[-1], download_root=download_root).cuda()
        self.layers = list(map(int, layers.split(',')))
        if self.layers[0] < 0:
            self.layers = None

    def read_audio(self, fname, sr=16e3):
        if isinstance(fname, str):
            audio = whisper.load_audio(fname)
        else:
            audio = fname
        audio_len = audio.shape[-1] // 320
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        return mel, audio_len

    def normalize(self, w):
        w = w.strip()
        w = ''.join([c for c in w if c.isalpha()])
        return w

    def get_features(self, loc):
        '''Return encoder features, word-level labels and segmentations'''
        # mel, audio_len = self.read_audio(loc)
        audio_len = loc.shape[1] 
        mel = loc.permute(0, 2, 1)

        layer_results = {}
        def get_layer_results(name):
            def hook(model, input, output):
                layer_results[name] = output.detach()
            return hook

        for l in self.layers:
            self.model.encoder.blocks[l].register_forward_hook(get_layer_results(f'block_{l}'))
    
        with torch.no_grad():
            if mel.ndim == 2:
                mel = mel[None]
            _ = self.model.encoder(mel)
            feats = []
            for l in self.layers:
                feat = layer_results[f'block_{l}'].contiguous() #XXX.squeeze(0)
                feats.append(feat[:, :audio_len])

        return torch.stack(feats, dim=-1)


class WhisperMLPClassifier(nn.Module):
    def __init__(self, whisper_model, embed_dim, depth, input_dim, layers=None):
        super().__init__()
        self.whisper = whisper_model
        self.layers = layers
        if layers is None:
            self.layers = self.whisper.layers
            
        self.weighted_sum = nn.Linear(len(self.layers), 1)
        dims = [input_dim]+[embed_dim]*depth
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(out_dim))
        layers.append(nn.Linear(embed_dim, 1))
        self.classifier = nn.Sequential(*layers)      

    def forward(self, x, size):
        if self.whisper is not None:
            feats = self.whisper.get_features(x)
            feats = feats.mean(1)
        else:
            feats = x

        feats = self.weighted_sum(feats).squeeze(-1)
        scores = self.classifier(feats).squeeze(-1)
        return scores

class BERTMLPClassifier(nn.Module):
    def __init__(self, bert_model, embed_dim, depth, input_dim, layers=None):
        super().__init__()
        self.bert = bert_model
        dims = [input_dim]+[embed_dim]*depth
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(out_dim))
        layers.append(nn.Linear(embed_dim, 1))
        self.classifier = nn.Sequential(*layers)      

    def forward(self, x, size):
        with torch.no_grad():
            outputs = self.bert(**x)
            
        feats = outputs.last_hidden_state
        attention_mask = x['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(feats.size()).float()
        sum_embeddings = torch.sum(feats * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask

        scores = self.classifier(embedding).squeeze(-1)
        return scores


class EfficientNetClassifier(nn.Module):
    def __init__(self, whisper_model, layers=None, model_id=0, n_class=1):
        super().__init__()
        self.whisper = whisper_model
        self.layers = layers
        self.weighted_sum = None
        if layers is not None:
            self.weighted_sum = nn.Linear(len(self.layers), 1)

        b = int(model_id)
        print('now train a effnet-b{:d} model'.format(b))
        if b == 7:
            self.model = torchvision.models.efficientnet_b7()
        elif b == 6:
            self.model = torchvision.models.efficientnet_b6()
        elif b == 5:
            self.model = torchvision.models.efficientnet_b5()
        elif b == 4:
            self.model = torchvision.models.efficientnet_b4()
        elif b == 3:
            self.model = torchvision.models.efficientnet_b3()
        elif b == 2:
            self.model = torchvision.models.efficientnet_b2()
        elif b == 1:
            self.model = torchvision.models.efficientnet_b1()
        elif b == 0:
            self.model = torchvision.models.efficientnet_b0()

        new_proj = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        print('conv1 get from pretrained model.')
        new_proj.weight = torch.nn.Parameter(torch.sum(self.model.features[0][0].weight, dim=1).unsqueeze(1))
        new_proj.bias = self.model.features[0][0].bias
        self.model.features[0][0] = new_proj
        self.model = create_feature_extractor(self.model, {'features.8': 'mout'})
        hid_dim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        self.cla = torch.nn.Sequential(nn.LayerNorm(hid_dim[int(model_id)]), nn.Linear(hid_dim[int(model_id)], n_class))

    def forward(self, x, valid_len):
        if self.weighted_sum is not None:
            if self.whisper is not None:
                feats = self.whisper.get_features(x)
            else:
                feats = x
            x = self.weighted_sum(feats).squeeze(-1)

        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.model(x)['mout']
        x_new = torch.stack([
            torch.mean(x[i, :, :, :math.ceil(valid_len[i] / 33)], dim=-1)
            for i in range(x.shape[0])
        ], dim=0)
        x = torch.mean(x_new, dim=[2])
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

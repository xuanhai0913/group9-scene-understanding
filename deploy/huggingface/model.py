from collections import OrderedDict

import torch
from torch import nn
import torchvision


class ConvRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activate=True,
        batch_norm=False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.activate = activate
        self.batch_norm = batch_norm

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.batch_norm:
            outputs = self.bn(outputs)
        return self.activation(outputs) if self.activate else outputs


class DecoderBlockResNet(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(
                middle_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.block(inputs)


class UnetResNet50(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, dropout=0.2):
        super().__init__()
        self.encoder = torchvision.models.resnet50(weights=None)
        filters = [2048, 2048, 1024, 512, 256]

        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=dropout)

        self.conv1 = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
            self.pool,
        )
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockResNet(
            filters[0],
            num_filters * 16,
            num_filters * 8,
        )
        self.dec5 = DecoderBlockResNet(
            filters[1] + num_filters * 8,
            num_filters * 16,
            num_filters * 8,
        )
        self.dec4 = DecoderBlockResNet(
            filters[2] + num_filters * 8,
            num_filters * 16,
            num_filters * 8,
        )
        self.dec3 = DecoderBlockResNet(
            filters[3] + num_filters * 8,
            num_filters * 8,
            num_filters * 2,
        )
        self.dec2 = DecoderBlockResNet(
            filters[4] + num_filters * 2,
            num_filters * 4,
            num_filters * 4,
        )
        self.dec1 = DecoderBlockResNet(
            num_filters * 4,
            num_filters * 4,
            num_filters,
        )
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.dropout(self.conv2(conv1))
        conv3 = self.dropout(self.conv3(conv2))
        conv4 = self.dropout(self.conv4(conv3))
        conv5 = self.dropout(self.conv5(conv4))

        center = self.center(self.pool(conv5))
        dec5 = self.dec5(torch.cat([center, conv5], dim=1))
        dec4 = self.dec4(torch.cat([dec5, conv4], dim=1))
        dec3 = self.dec3(torch.cat([dec4, conv3], dim=1))
        dec2 = self.dropout(self.dec2(torch.cat([dec3, conv2], dim=1)))
        dec1 = self.dec1(dec2)
        return self.final(self.dec0(dec1))


def load_road_model(checkpoint_path):
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = OrderedDict(
            (key.removeprefix("module."), value)
            for key, value in state_dict.items()
        )

    num_classes = state_dict["final.weight"].shape[0]
    model = UnetResNet50(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model

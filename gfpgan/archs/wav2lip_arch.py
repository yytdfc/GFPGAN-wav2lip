import math
import random
import torch
from basicsr.archs.stylegan2_arch import (
    ConvLayer,
    EqualConv2d,
    EqualLinear,
    ResBlock,
    ScaledLeakyReLU,
    StyleGAN2Generator,
)
from basicsr.ops.fused_act import FusedLeakyReLU
from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn
from torch.nn import functional as F

# architecture from https://github.com/Rudrabha/Wav2Lip


class Conv2d(nn.Module):
    def __init__(
        self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding), nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class nonorm_Conv2d(nn.Module):
    def __init__(
        self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


class Conv2dTranspose(nn.Module):
    def __init__(
        self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


@ARCH_REGISTRY.register()
class Wav2Lip(nn.Module):
    def __init__(self, output_block="raw"):
        super(Wav2Lip, self).__init__()
        self.output_block = output_block

        self.face_encoder_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2d(6, 16, kernel_size=7, stride=1, padding=3)
                ),  # 96,96
                nn.Sequential(
                    Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 48,48
                    Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 24,24
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 12,12
                    Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 6,6
                    Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 3,3
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1, 1
                    Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                ),
            ]
        )

        self.audio_encoder = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            Conv2d(1, 32, kernel_size=(5, 7), stride=(2, 4), padding=(2, 3)),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            Conv2d(32, 64, kernel_size=(5, 7), stride=(2, 4), padding=(2, 3)),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            Conv2d(64, 128, kernel_size=(3, 7), stride=(1, 4), padding=(1, 3)),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            nn.Linear(16, 16),
            nn.LayerNorm(16),
            Conv2d(128, 256, kernel_size=(3, 7), stride=(1, 4), padding=(1, 3)),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            nn.Linear(4, 4),
            nn.LayerNorm(4),
            Conv2d(256, 512, kernel_size=(3, 4), stride=(1, 1), padding=(0, 0)),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

        self.face_decoder_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                ),
                nn.Sequential(
                    Conv2dTranspose(
                        1024, 512, kernel_size=3, stride=1, padding=0
                    ),  # 3,3
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2dTranspose(
                        1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 6, 6
                nn.Sequential(
                    Conv2dTranspose(
                        768, 384, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 12, 12
                nn.Sequential(
                    Conv2dTranspose(
                        512, 256, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 24, 24
                nn.Sequential(
                    Conv2dTranspose(
                        320, 128, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 48, 48
                nn.Sequential(
                    Conv2dTranspose(
                        160, 64, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                ),
            ]
        )  # 96,96

        if self.output_block == "add_in":
            self.output_block = nn.Sequential(
                Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid(),
            )
        else:
            self.output_block = nn.Sequential(
                Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid(),
            )

    def forward(self, audio_sequences, face_sequences):
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())

        audio_embedding = self.audio_encoder(audio_sequences)  # B, 512, 1, 1

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding.expand(-1, -1, x.size(2), x.size(3))
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e

            feats.pop()

        x = self.output_block(x)
        if self.output_block == "add_in":
            alpha = x[:, 3:4, :, :]
            # x = x[:, :3, :, :] * alpha + face_sequences * (1 - alpha)
            x = (x[:, :3, :, :] - face_sequences) * alpha + face_sequences

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)  # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2)  # (B, C, T, H, W)

        else:
            outputs = x

        return outputs


if __name__ == "__main__":
    model = Wav2Lip(output_block="add_in")
    with torch.no_grad():
        audio_sequences = torch.randn(2, 1, 10, 1024)
        face_sequences = torch.randn(2, 6, 256, 256)
        outputs = model(audio_sequences, face_sequences)
        print(outputs.size())
        # print(outputs)
        print("Wav2Lip model is working!")

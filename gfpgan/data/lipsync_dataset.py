import math
import re
import os.path as osp
import glob
import cv2
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data

import imgaug.augmenters as iaa

from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class LipSyncDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.dataroot = opt["dataroot"]
        self.turn_to_simple_aug = int(opt.get("turn_to_simple_aug", -1))
        self.videos = opt.get("videos", None)
        if not self.videos:
            self.videos = glob.glob(f"{opt['dataroot']}/*")

        self.frames = {}
        for video in self.videos:
            frames = glob.glob(f"{video}/*.webp")
            frames.sort(key=self.get_frame_id)
            if frames:
                self.frames[video] = frames
        self.videos = list(self.frames.keys())

        self.audios = {
            video: np.load(f"{video}.npy", allow_pickle=True).transpose((0, 2, 1))
            for video in self.videos
        }
        print(self.audios[self.videos[0]].shape)

        self.out_size = opt["out_size"]
        self.input_frames_num = opt["input_frames_num"]
        self.audio_window_length = opt["audio_window_length"]
        self.audio_window_before = opt["audio_window_before"]
        self.color_transform = iaa.Sequential(
            [
                # color
                iaa.MultiplyAndAddToBrightness(mul=(0.9, 1.1), add=(-10, 10)),
                iaa.GammaContrast((0.5, 2.0)),
                iaa.Sometimes(0.15, iaa.pillike.EnhanceSharpness(factor=[0.8, 1.2])),
                iaa.Sometimes(0.05, iaa.pillike.EnhanceColor(factor=[0.8, 1.2])),
            ],
            random_order=True,
        )
        self.global_step = 0

    def __len__(self):
        return sum(map(len, self.frames.values()))

    def get_frame_id(self, frame):
        return int(re.findall(r"\d+", frame)[-1])

    def perspective_transform(self, img, source_coords, target_size):
        w, h = target_size
        target_coords = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
        matrix = []
        for s, t in zip(source_coords, target_coords):
            matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0] * t[0], -s[0] * t[1]])
            matrix.append([0, 0, 0, t[0], t[1], 1, -s[1] * t[0], -s[1] * t[1]])
        A = np.matrix(matrix, dtype=np.float32)
        B = np.array(source_coords).reshape(8)
        coeffs = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        coeffs = np.array(coeffs).reshape(8)
        m = np.append(coeffs, 1).reshape((3, 3)).copy()
        img = img.transform(target_size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
        return img

    def __getitem__(self, idx):
        video = self.videos[idx % len(self.videos)]

        def random_consecutive_frames_audios(frames, audio_feature):
            start_frame = np.random.randint(
                self.audio_window_length // 4, len(frames) - self.input_frames_num
            )
            last_frame = start_frame + self.input_frames_num - 1
            half_window = self.audio_window_length // 2
            if (
                last_frame * 2 + self.audio_window_length - self.audio_window_before
                > audio_feature.shape[2]
                or last_frame * 2 - self.audio_window_before < 0
            ):
                return random_consecutive_frames_audios(frames, audio_feature)
            # check if these frames are consecutive
            start_frame = int(start_frame)
            for i in range(self.input_frames_num - 1):
                if (
                    self.get_frame_id(frames[start_frame + i + 1])
                    - self.get_frame_id(frames[start_frame + i])
                ) != 1:
                    return random_consecutive_frames_audios(frames, audio_feature)
            return (
                frames[start_frame : start_frame + self.input_frames_num],
                audio_feature[
                    :,
                    :,
                    last_frame * 2
                    - self.audio_window_before : last_frame * 2
                    + (self.audio_window_length - self.audio_window_before),
                ],
            )

        input_frames, input_audio = random_consecutive_frames_audios(
            self.frames[video], self.audios[video]
        )

        input_frames = [Image.open(frame).convert("RGB") for frame in input_frames]
        w, h = input_frames[0].size

        if self.turn_to_simple_aug >= 0 and self.global_step >= self.turn_to_simple_aug:
            pass
        else:
            shift_x = round(w * 0.05)
            shift_y = round(h * 0.05)
            shift = (
                np.concatenate(
                    [
                        np.random.randint(-shift_x, shift_x, 4),
                        np.random.randint(-shift_y, shift_y, 4),
                    ]
                )
                .reshape(2, 4)
                .transpose()
            )
            input_frames = [
                self.perspective_transform(
                    frame,
                    np.array(
                        [
                            (0, 0),
                            (frame.width, 0),
                            (frame.width, frame.height),
                            (0, frame.height),
                        ]
                    )
                    + shift,
                    (frame.width, frame.height),
                )
                for frame in input_frames
            ]
            deterministic_transform = self.color_transform.to_deterministic()
            input_frames = [
                deterministic_transform(image=np.array(frame)) for frame in input_frames
            ]

        input_frames = (
            torch.from_numpy(np.concatenate(input_frames, 2)).float().permute(2, 0, 1)
        )
        input_frames = input_frames / 255.0
        gt = input_frames[-3:].clone()
        input_frames[
            -3:, round(h * 0.60) : round(h * 0.96), round(w * 0.25) : -round(w * 0.25)
        ] = 0

        cut = 112 / 256 / 2
        input_frames = input_frames[
            :, round(h * cut * 2) : round(h), round(w * cut) : -round(w * cut)
        ]
        gt = gt[:, round(h * cut * 2) : round(h), round(w * cut) : -round(w * cut)]

        if np.random.randint(2):
            input_frames[:3] = 0

        input_frames = torch.nn.functional.interpolate(
            input_frames.unsqueeze(0),
            (self.out_size, self.out_size),
            mode="bicubic",
        ).squeeze(0)
        gt = torch.nn.functional.interpolate(
            gt.unsqueeze(0),
            (self.out_size, self.out_size),
            mode="bicubic",
        ).squeeze(0)

        self.global_step += 1

        return {
            "frames": input_frames,
            "gt": gt,
            "audio": torch.from_numpy(input_audio.transpose(0, 2, 1)).float(),
        }


if __name__ == "__main__":
    dataset = LipSyncDataset(
        {
            "dataroot": "facedata/",
            "out_size": 256,
            "videos": [],
            #  "mean": [0.5, 0.5, 0.5],
            #  "std": [0.5, 0.5, 0.5],
            "input_frames_num": 2,
            "audio_window_length": 10,
            "audio_window_before": 2,
            "turn_to_simple_aug": 1,
        }
    )
    print(len(dataset))
    import fastcv

    for _ in range(10):
        item = dataset[0]
        fastcv.view(
            fastcv.cast(item["frames"][:3], "pil", from_range=[0, 1]),
            fastcv.cast(item["frames"][3:], "pil", from_range=[0, 1]),
            fastcv.cast(item["gt"], "pil", from_range=[0, 1]),
        )
        print(item["frames"].shape)
        print(item["audio"].shape)

    from gfpgan.archs.wav2lip_arch import Wav2Lip

    wav2lip = Wav2Lip()
    ckpt = torch.load("experiments/train_lipsync_simple/models/net_g_66000.pth")
    wav2lip.load_state_dict(ckpt["params"])
    wav2lip.eval()

    with torch.no_grad():
        item["audio"].fill_(0)
        pred = wav2lip(item["audio"].unsqueeze(0), item["frames"].unsqueeze(0))
    print(pred.shape, pred.min(), pred.max())
    print(item["frames"].shape, item["frames"].min(), item["frames"].max())
    fastcv.view(fastcv.cast(pred[0], "pil", from_range=[0, 1]))

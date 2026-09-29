# Copyright (c) 2025, Infineon Technologies AG, or an affiliate of Infineon Technologies AG. All rights reserved.
# This software, associated documentation and materials ("Software") is owned by Infineon Technologies AG or one
# of its affiliates ("Infineon") and is protected by and subject to worldwide patent protection, worldwide copyright laws,
# and international treaty provisions. Therefore, you may use this Software only as provided in the license agreement accompanying
# the software package from which you obtained this Software. If no license agreement applies, then any use, reproduction, modification,
# translation, or compilation of this Software is prohibited without the express written permission of Infineon.
# Disclaimer: UNLESS OTHERWISE EXPRESSLY AGREED WITH INFINEON, THIS SOFTWARE IS PROVIDED AS-IS, WITH NO WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, ALL WARRANTIES OF NON-INFRINGEMENT OF THIRD-PARTY RIGHTS AND IMPLIED WARRANTIES
# SUCH AS WARRANTIES OF FITNESS FOR A SPECIFIC USE/PURPOSE OR MERCHANTABILITY. Infineon reserves the right to make changes to the Software
# without notice. You are responsible for properly designing, programming, and testing the functionality and safety of your intended application
# of the Software, as well as complying with any legal requirements related to its use. Infineon does not guarantee that the Software will be
# free from intrusion, data theft or loss, or other breaches ("Security Breaches"), and Infineon shall have no liability arising out of any
# Security Breaches. Unless otherwise explicitly approved by Infineon, the Software may not be used in any application where a failure of the
# Product or any consequences of the use thereof can reasonably be expected to result in personal injury.

# ============================================
# IMPORTS
# ============================================
import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# ============================================
# SET GLOBAL SEEDS FOR REPRODUCIBILITY
# ============================================
torch.manual_seed(42)
np.random.seed(42)

# ============================================
# SUPPRESS WARNINGS
# ============================================
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)

# ============================================
# CONFIG
# ============================================
BASE_DIR = os.path.dirname(__file__)
CARLA_ROOT = os.path.join(BASE_DIR, "data", "carla")
SAMPLE_BBOX_IMAGE = os.path.join(
    BASE_DIR, "data", "carla", "Town02", "pos0", "bbox_images", "000001.png"
)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "exported_model", "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "exported_model", "results")
RNN_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "rnn_real_best.pth")

# Image
IMG_W = 1280
IMG_H = 720
NORM_SHIFT = 0.5

# Classes
CLASSES = {
    0: "bicycle",
    1: "car",
    2: "motorcycle",
    3: "bus",
    4: "truck",
}

# RNN params
INPUT_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 1
OUTPUT_SIZE = 4

# LSTM params
LSTM_MODE = "aurix"
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 1
MAX_TRACKS = 10
MAX_DETS = 15
LSTM_INPUT_SIZE = MAX_DETS
LSTM_FORGET_BIAS = 1.0

# RNN Training
BATCH_SIZE = 10
LEARNING_RATE = 0.0003
LR_DECAY = 0.05
LR_DECAY_STEP = 20000
MAX_ITERS = 200000
SEQ_LENGTH = 20

# LSTM Training
LSTM_BATCH_SIZE = 10
LSTM_LEARNING_RATE = 0.001
LSTM_LR_DECAY = 0.05
LSTM_LR_DECAY_STEP = 20000
LSTM_MAX_ITERS = 200000
LSTM_SEQ_LENGTH = 20

# Loss weights
LOSS_LAMBDA = 1.0
LOSS_KAPPA = 1.0
LOSS_NU = 1.0
LOSS_XI = 0.1

# Tracker
EXIST_THRESHOLD = 0.6
IOU_THRESHOLD = 0.3
MAX_AGE = 5
MIN_HITS = 2

# Town splits
TRAIN_TOWNS = [
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town06",
    "Town10HD",
]
VAL_TOWNS = ["Town05"]
TEST_TOWNS = ["Town07"]
POSITIONS = ["pos0", "pos1", "pos2"]


# ============================================
# BBOX UTILITIES
# Milan et al. 2017
# "normalised to [-0.5, 0.5]
#  w.r.t. the image dimensions"
# ============================================
def normalize_bbox(bbox):
    """
    Normalize bbox to [-0.5, 0.5].
    Args:
        bbox: [x, y, w, h] in pixels
    Returns:
        [x_n, y_n, w_n, h_n] in [-0.5, 0.5]
    """
    x, y, w, h = bbox
    return [
        x / IMG_W - NORM_SHIFT,
        y / IMG_H - NORM_SHIFT,
        w / IMG_W - NORM_SHIFT,
        h / IMG_H - NORM_SHIFT,
    ]


def denormalize_bbox(bbox):
    """
    Inverse of normalize_bbox.
    Args:
        bbox: [x_n, y_n, w_n, h_n]
    Returns:
        [x, y, w, h] in pixels
    """
    x, y, w, h = bbox
    return [
        (x + NORM_SHIFT) * IMG_W,
        (y + NORM_SHIFT) * IMG_H,
        (w + NORM_SHIFT) * IMG_W,
        (h + NORM_SHIFT) * IMG_H,
    ]


def clip_bbox(bbox):
    """Clip bbox to image boundaries."""
    x, y, w, h = bbox
    x = max(0.0, min(float(x), IMG_W - 1))
    y = max(0.0, min(float(y), IMG_H - 1))
    w = max(1.0, min(float(w), IMG_W - x))
    h = max(1.0, min(float(h), IMG_H - y))
    return [x, y, w, h]


def xywh_to_xyxy(bbox):
    """[x, y, w, h] to [x1, y1, x2, y2]"""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def xyxy_to_xywh(bbox):
    """[x1, y1, x2, y2] to [x, y, w, h]"""
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def euclidean_dist_sq(b1, b2):
    """Squared Euclidean distance."""
    return sum((a - b) ** 2 for a, b in zip(b1, b2))


def _normalize_bbox_local(bbox):
    """Normalize bbox locally."""
    x, y, w, h = bbox
    return [
        x / IMG_W - NORM_SHIFT,
        y / IMG_H - NORM_SHIFT,
        w / IMG_W - NORM_SHIFT,
        h / IMG_H - NORM_SHIFT,
    ]


# ============================================
# PRIVATE DATA HELPERS
# ============================================
def _load_tracks_from_gt(gt_txt_path, min_len=10):
    """Load tracks from gt.txt file."""
    tracks = defaultdict(list)
    if not os.path.exists(gt_txt_path):
        return {}
    with open(gt_txt_path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frame = int(p[0])
            track_id = int(p[1])
            x = float(p[2])
            y = float(p[3])
            w = float(p[4])
            h = float(p[5])
            tracks[track_id].append((frame, x, y, w, h))
    result = {}
    for tid, frames in tracks.items():
        frames_sorted = sorted(frames, key=lambda x: x[0])
        if len(frames_sorted) >= min_len:
            result[tid] = frames_sorted
    return result


def _load_multi_target_frames(gt_txt_path, min_tracks=2):
    """Load frames with multiple tracks."""
    if not os.path.exists(gt_txt_path):
        return []
    frame_tracks = defaultdict(dict)
    with open(gt_txt_path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frame = int(p[0])
            track_id = int(p[1])
            bbox = [
                float(p[2]),
                float(p[3]),
                float(p[4]),
                float(p[5]),
            ]
            frame_tracks[frame][track_id] = bbox
    frames = []
    for frame_id in sorted(frame_tracks.keys()):
        tracks = frame_tracks[frame_id]
        if len(tracks) >= min_tracks:
            frames.append(
                {
                    "frame_id": frame_id,
                    "tracks": tracks,
                }
            )
    return frames


# ============================================
# RNN MOTION MODEL
# Milan et al. 2017 - Figure 2 (left)
# ============================================
class RNNMotionModel(nn.Module):
    """
    RNN for state prediction, update,
    and existence probability estimation.
    Milan et al. 2017 Figure 2 (left).
    hidden_size=64 for AURIX TC4Dx.
    """

    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
    ):
        super(RNNMotionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.pred_rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )
        self.update_rnn = nn.RNN(
            input_size=input_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.update_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, output_size),
        )
        self.exist_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x,
        measurement=None,
        assignment=None,
        pred_hidden=None,
        upd_hidden=None,
    ):
        rnn_out, pred_hidden = self.pred_rnn(x, pred_hidden)
        pred = self.pred_head(rnn_out)
        exist = self.exist_head(rnn_out)
        if measurement is not None:
            if assignment is not None:
                weighted = assignment * measurement + (1.0 - assignment) * pred
            else:
                weighted = measurement
            update_input = torch.cat([pred, weighted], dim=-1)
            upd_out, upd_hidden = self.update_rnn(update_input, upd_hidden)
            updated = self.update_head(upd_out)
        else:
            updated = pred
            upd_hidden = upd_hidden
        return (
            pred,
            updated,
            exist,
            pred_hidden,
            upd_hidden,
        )

    def init_hidden(self, batch_size=1):
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
        )

    def predict_next(self, bbox, pred_hidden, device):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(
                [[bbox]],
                dtype=torch.float32,
            ).to(device)
            rnn_out, new_hidden = self.pred_rnn(x, pred_hidden)
            pred = self.pred_head(rnn_out)
            exist = self.exist_head(rnn_out)
            pred_bbox = pred.squeeze(0).squeeze(0).cpu().tolist()
            exist_prob = exist.squeeze().cpu().item()
        return pred_bbox, exist_prob, new_hidden

    def update_state(
        self,
        pred_bbox,
        meas_bbox,
        assignment,
        upd_hidden,
        device,
    ):
        self.eval()
        with torch.no_grad():
            pred = torch.tensor(
                [[pred_bbox]],
                dtype=torch.float32,
            ).to(device)
            meas = torch.tensor(
                [[meas_bbox]],
                dtype=torch.float32,
            ).to(device)
            a = torch.tensor(
                [[[assignment]]],
                dtype=torch.float32,
            ).to(device)
            weighted = a * meas + (1.0 - a) * pred
            update_input = torch.cat([pred, weighted], dim=-1)
            upd_out, new_upd_hidden = self.update_rnn(update_input, upd_hidden)
            updated = self.update_head(upd_out)
            updated_bbox = updated.squeeze(0).squeeze(0).cpu().tolist()
        return updated_bbox, new_upd_hidden


# ============================================
# LSTM ASSOCIATION MODEL
# Milan et al. 2017 - Figure 2 (right)
# ============================================
class LSTMAssociation(nn.Module):
    """
    LSTM for data association.
    Milan et al. 2017 Figure 2 (right).
    Three domain adaptations for CARLA:
    1. Row-based input
    2. Rank-based distances
    3. Loss normalization
    """

    def __init__(
        self,
        input_size=LSTM_INPUT_SIZE,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        max_dets=MAX_DETS,
        forget_bias=LSTM_FORGET_BIAS,
    ):
        super(LSTMAssociation, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_dets = max_dets
        self.forget_bias = forget_bias

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_head = nn.Linear(hidden_size, max_dets)
        self._init_forget_bias()

    def _init_forget_bias(self):
        for layer in range(self.num_layers):
            bias_ih = getattr(self.lstm, "bias_ih_l{}".format(layer))
            bias_hh = getattr(self.lstm, "bias_hh_l{}".format(layer))
            H = self.hidden_size
            with torch.no_grad():
                bias_ih.data[H : 2 * H].fill_(self.forget_bias)
                bias_hh.data[H : 2 * H].fill_(self.forget_bias)

    def init_hidden(self, batch_size=1):
        device = next(self.parameters()).device
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.shape[0])
        out, hidden = self.lstm(x, hidden)
        logits = self.output_head(out.squeeze(1))
        probs = torch.softmax(logits, dim=-1)
        return probs, hidden

    def build_distance_matrix(self, pred_bboxes, det_bboxes, n_tracks, n_dets, device):
        dist = torch.zeros(n_tracks, n_dets, device=device)
        for i in range(n_tracks):
            xi = torch.tensor(
                pred_bboxes[i],
                dtype=torch.float32,
                device=device,
            )
            for j in range(n_dets):
                zj = torch.tensor(
                    det_bboxes[j],
                    dtype=torch.float32,
                    device=device,
                )
                dist[i, j] = torch.sum((xi - zj) ** 2)
        C = torch.full(
            (MAX_TRACKS, MAX_DETS),
            fill_value=1.0,
            dtype=torch.float32,
            device=device,
        )
        for i in range(n_tracks):
            order = dist[i].argsort()
            for rank, idx in enumerate(order):
                C[i, idx] = rank / max(n_dets - 1, 1)
        return C

    def associate(self, pred_bboxes, det_bboxes, n_tracks, n_dets, device):
        if n_tracks == 0 and n_dets == 0:
            return [], [], []
        if n_tracks == 0:
            return [], [], list(range(n_dets))
        if n_dets == 0:
            return [], list(range(n_tracks)), []
        self.eval()
        with torch.no_grad():
            C = self.build_distance_matrix(
                pred_bboxes, det_bboxes, n_tracks, n_dets, device
            )
            hidden = self.init_hidden(1)
            matched = []
            assigned_d = set()
            for i in range(n_tracks):
                row_i = C[i].unsqueeze(0).unsqueeze(0)
                probs, hidden = self.forward(row_i, hidden)
                valid = probs[0, :n_dets].clone()
                for d in assigned_d:
                    valid[d] = 0.0
                best_det = valid.argmax().item()
                best_prob = valid[best_det].item()
                threshold = 0.0 if n_dets == 1 else 1.0 / (n_dets * 3)
                if best_prob > threshold:
                    matched.append((i, best_det))
                    assigned_d.add(best_det)
        matched_t = {t for t, d in matched}
        matched_d = {d for t, d in matched}
        unmatched_t = [i for i in range(n_tracks) if i not in matched_t]
        unmatched_d = [j for j in range(n_dets) if j not in matched_d]
        return matched, unmatched_t, unmatched_d


# ============================================
# EXPORT WRAPPER MODELS
# ============================================
class RNNPredictionOnly(nn.Module):
    """
    Wrapper for RNN ONNX export.
    Returns only prediction tensor.
    """

    def __init__(self, rnn_model):
        super(RNNPredictionOnly, self).__init__()
        self.rnn = rnn_model

    def forward(self, x):
        pred, _, _, _, _ = self.rnn(x)
        return pred


class LSTMProbsOnly(nn.Module):
    """
    Wrapper for LSTM ONNX export.
    Returns only probability tensor.
    """

    def __init__(self, lstm_model):
        super(LSTMProbsOnly, self).__init__()
        self.lstm = lstm_model

    def forward(self, x):
        probs, _ = self.lstm(x)
        return probs


# ============================================
# RNN DATASET
# ============================================
class RealTrajectoryDataset(Dataset):
    """
    Dataset from real CARLA gt.txt files.
    Returns SEQ_LENGTH windows of trajectories.
    """

    def __init__(
        self,
        split="train",
        seq_length=SEQ_LENGTH,
        min_len=10,
        stride=5,
        carla_root=None,
    ):
        self.seq_length = seq_length
        self.sequences = []
        if carla_root is None:
            carla_root = CARLA_ROOT
        towns = (
            TRAIN_TOWNS
            if split == "train"
            else VAL_TOWNS if split == "val" else TEST_TOWNS
        )
        print("  Loading trajectories " "({})...".format(split))
        total_tracks = 0
        total_seqs = 0
        for town in towns:
            for pos in POSITIONS:
                gt_path = os.path.join(
                    carla_root,
                    town,
                    pos,
                    "gt.txt",
                )
                if not os.path.exists(gt_path):
                    continue
                tracks = _load_tracks_from_gt(gt_path, min_len=min_len)
                total_tracks += len(tracks)
                for tid, frames in tracks.items():
                    win_size = seq_length + 1
                    n_frames = len(frames)
                    for start in range(
                        0,
                        n_frames - win_size + 1,
                        stride,
                    ):
                        window = frames[start : start + win_size]
                        bboxes = []
                        exists = []
                        prev_frame = None
                        for f, x, y, w, h in window:
                            bbox = [x, y, w, h]
                            norm = normalize_bbox(bbox)
                            bboxes.append(norm)
                            if prev_frame is not None:
                                gap = f - prev_frame
                                alive = gap == 1
                            else:
                                alive = True
                            exists.append(1.0 if alive else 0.0)
                            prev_frame = f
                        self.sequences.append((bboxes, exists))
                        total_seqs += 1
        print("  Tracks: {:,} | " "Sequences: {:,}".format(total_tracks, total_seqs))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        bboxes, exists = self.sequences[idx]
        bbox_t = torch.tensor(bboxes, dtype=torch.float32)
        exist_t = torch.tensor(exists, dtype=torch.float32).unsqueeze(-1)
        return (
            bbox_t[:-1],
            bbox_t[1:],
            exist_t[:-1],
            exist_t[1:],
        )


# ============================================
# LSTM DATASET
# ============================================
class LSTMAssociationDataset(Dataset):
    """
    Dataset for LSTM association training.
    Each sample = one frame with N tracks,
    M detections and distance matrix C.
    """

    def __init__(
        self,
        split="train",
        n_samples=100000,
        min_tracks=2,
        max_noise=0.02,
        clutter_pct=0.3,
        carla_root=None,
    ):
        self.max_noise = max_noise
        self.clutter_pct = clutter_pct
        self.samples = []
        if carla_root is None:
            carla_root = CARLA_ROOT
        towns = TRAIN_TOWNS if split == "train" else VAL_TOWNS
        print("  Loading LSTM frames " "({})...".format(split))
        all_frames = []
        for town in towns:
            for pos in POSITIONS:
                gt_path = os.path.join(
                    carla_root,
                    town,
                    pos,
                    "gt.txt",
                )
                frames = _load_multi_target_frames(gt_path, min_tracks=min_tracks)
                all_frames.extend(frames)
        print("  Multi-track frames: {:,}".format(len(all_frames)))
        rng = np.random.RandomState(42)
        for i in range(n_samples):
            if all_frames:
                frame = all_frames[i % len(all_frames)]
                sample = self._make_sample(frame, rng)
            else:
                sample = self._make_synthetic(rng)
            if sample is not None:
                self.samples.append(sample)
        print("  Samples: {:,}".format(len(self.samples)))

    def _make_sample(self, frame, rng):
        tracks = frame["tracks"]
        track_ids = list(tracks.keys())
        if len(track_ids) > MAX_TRACKS:
            track_ids = track_ids[:MAX_TRACKS]
        n_tracks = rng.randint(2, MAX_TRACKS + 1)
        real_bboxes = [_normalize_bbox_local(tracks[tid]) for tid in track_ids]
        gt_bboxes = []
        for bbox in real_bboxes:
            if len(gt_bboxes) >= n_tracks:
                break
            gt_bboxes.append(list(bbox))
        while len(gt_bboxes) < n_tracks:
            gt_bboxes.append(
                [
                    float(rng.uniform(-0.45, 0.45)),
                    float(rng.uniform(-0.45, 0.45)),
                    float(rng.uniform(-0.48, -0.35)),
                    float(rng.uniform(-0.48, -0.38)),
                ]
            )
        pred_bboxes = [
            [b + rng.normal(0, self.max_noise) for b in bbox] for bbox in gt_bboxes
        ]
        n_clutter = int(n_tracks * self.clutter_pct)
        n_dets = min(n_tracks + n_clutter, MAX_DETS)
        det_order = list(range(n_tracks))
        rng.shuffle(det_order)
        det_bboxes = [
            [b + rng.normal(0, self.max_noise * 0.5) for b in gt_bboxes[idx]]
            for idx in det_order
        ]
        for _ in range(n_clutter):
            det_bboxes.append(
                [
                    rng.uniform(-0.5, 0.5),
                    rng.uniform(-0.5, 0.5),
                    rng.uniform(-0.5, 0.0),
                    rng.uniform(-0.5, 0.0),
                ]
            )
        labels = []
        for i in range(n_tracks):
            if i in det_order:
                labels.append(det_order.index(i))
            else:
                labels.append(-1)
        dist_matrix = torch.zeros(n_tracks, n_dets)
        for i in range(n_tracks):
            for j in range(n_dets):
                dist_matrix[i, j] = euclidean_dist_sq(
                    pred_bboxes[i],
                    det_bboxes[j],
                )
        C = torch.full(
            (MAX_TRACKS, MAX_DETS),
            fill_value=1.0,
            dtype=torch.float32,
        )
        for i in range(n_tracks):
            row = dist_matrix[i]
            order = row.argsort()
            for rank, idx in enumerate(order):
                C[i, idx] = rank / max(n_dets - 1, 1)
        C_flat = C.view(-1)
        labels_padded = torch.full(
            (MAX_TRACKS,),
            -1,
            dtype=torch.long,
        )
        labels_padded[:n_tracks] = torch.tensor(labels, dtype=torch.long)
        return {
            "C_flat": C_flat,
            "labels": labels_padded,
            "n_tracks": n_tracks,
            "n_dets": n_dets,
        }

    def _make_synthetic(self, rng):
        n_tracks = rng.randint(2, MAX_TRACKS + 1)
        n_dets = min(
            n_tracks + rng.randint(0, 3),
            MAX_DETS,
        )
        pred_bboxes = [
            [
                rng.uniform(-0.4, 0.4),
                rng.uniform(-0.3, 0.3),
                rng.uniform(-0.5, -0.3),
                rng.uniform(-0.5, -0.35),
            ]
            for _ in range(n_tracks)
        ]
        det_order = list(range(n_tracks))
        rng.shuffle(det_order)
        det_bboxes = [
            [pred_bboxes[idx][k] + rng.normal(0, 0.01) for k in range(4)]
            for idx in det_order
        ]
        for _ in range(n_dets - n_tracks):
            det_bboxes.append(
                [
                    rng.uniform(-0.5, 0.5),
                    rng.uniform(-0.5, 0.5),
                    rng.uniform(-0.5, 0.0),
                    rng.uniform(-0.5, 0.0),
                ]
            )
        dist_matrix = torch.zeros(n_tracks, n_dets)
        for i in range(n_tracks):
            for j in range(n_dets):
                dist_matrix[i, j] = euclidean_dist_sq(
                    pred_bboxes[i],
                    det_bboxes[j],
                )
        C = torch.full(
            (MAX_TRACKS, MAX_DETS),
            fill_value=1.0,
            dtype=torch.float32,
        )
        for i in range(n_tracks):
            row = dist_matrix[i]
            order = row.argsort()
            for rank, idx in enumerate(order):
                C[i, idx] = rank / max(n_dets - 1, 1)
        C_flat = C.view(-1)
        labels = []
        for i in range(n_tracks):
            if i in det_order:
                labels.append(det_order.index(i))
            else:
                labels.append(-1)
        labels_padded = torch.full(
            (MAX_TRACKS,),
            -1,
            dtype=torch.long,
        )
        labels_padded[:n_tracks] = torch.tensor(labels, dtype=torch.long)
        return {
            "C_flat": C_flat,
            "labels": labels_padded,
            "n_tracks": n_tracks,
            "n_dets": n_dets,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s["C_flat"],
            s["labels"],
            s["n_tracks"],
            s["n_dets"],
        )


# ============================================
# RNN LOSS
# Milan et al. 2017 - Equation 3
# ============================================
class TrackingLoss(nn.Module):
    """
    Combined tracking loss.
    L = lambda*pred_MSE + kappa*update_MSE
      + nu*exist_BCE + xi*smoothness
    """

    def __init__(
        self,
        lambda_=LOSS_LAMBDA,
        kappa=LOSS_KAPPA,
        nu=LOSS_NU,
        xi=LOSS_XI,
    ):
        super(TrackingLoss, self).__init__()
        self.lambda_ = lambda_
        self.kappa = kappa
        self.nu = nu
        self.xi = xi
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(
        self,
        pred,
        updated,
        exist,
        gt_bbox,
        gt_exist,
    ):
        pred_loss = self.mse(pred, gt_bbox)
        update_loss = self.mse(updated, gt_bbox)
        exist_c = exist.clamp(1e-7, 1.0 - 1e-7)
        exist_loss = self.bce(exist_c, gt_exist)
        if exist.shape[1] > 1:
            smooth_loss = torch.mean(torch.abs(exist[:, 1:, :] - exist[:, :-1, :]))
        else:
            smooth_loss = torch.tensor(0.0, device=exist.device)
        total = (
            self.lambda_ * pred_loss
            + self.kappa * update_loss
            + self.nu * exist_loss
            + self.xi * smooth_loss
        )
        details = {
            "pred_loss": pred_loss.item(),
            "update_loss": update_loss.item(),
            "exist_loss": exist_loss.item(),
            "smooth_loss": smooth_loss.item(),
            "total": total.item(),
        }
        return total, details


# ============================================
# LSTM LOSS
# Milan et al. 2017 - Equation 5
# ============================================
class LSTMAssociationLoss(nn.Module):
    """
    NLL loss for LSTM data association.
    L(A^i, a~) = -log(A^i_a~)
    """

    def __init__(self):
        super(LSTMAssociationLoss, self).__init__()

    def forward(
        self,
        all_probs,
        labels,
        n_tracks_batch,
        n_dets_batch=None,
    ):
        B = all_probs.shape[0]
        total_loss = 0.0
        n_valid = 0
        correct = 0
        total_preds = 0
        for b in range(B):
            n_t = n_tracks_batch[b].item()
            n_d = int(n_dets_batch[b].item()) if n_dets_batch is not None else MAX_DETS
            for i in range(n_t):
                label = labels[b, i].item()
                if label < 0 or label >= n_d:
                    continue
                probs_i = all_probs[b, i]
                probs_valid = probs_i[:n_d]
                probs_sum = probs_valid.sum()
                if probs_sum > 1e-8:
                    probs_valid = probs_valid / probs_sum
                else:
                    probs_valid = (
                        torch.ones(
                            n_d,
                            device=probs_i.device,
                        )
                        / n_d
                    )
                prob_correct = probs_valid[label].clamp(1e-7, 1.0)
                total_loss += -torch.log(prob_correct)
                n_valid += 1
                if probs_valid.argmax().item() == label:
                    correct += 1
                total_preds += 1
        if n_valid == 0:
            loss = torch.tensor(
                0.0,
                requires_grad=True,
                device=all_probs.device,
            )
        else:
            loss = total_loss / n_valid
        accuracy = correct / total_preds * 100 if total_preds > 0 else 0.0
        details = {
            "loss": (loss.item() if hasattr(loss, "item") else float(loss)),
            "n_valid": n_valid,
            "accuracy_pct": accuracy,
        }
        return loss, details


# ============================================
# PUBLIC API
# ============================================
def get_model(model_type):
    """
    Get RNN or LSTM model.
    Args:
        model_type: 'rnn' or 'lstm'
    Returns:
        RNNMotionModel or LSTMAssociation
    """
    if model_type == "rnn":
        return RNNMotionModel(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            output_size=OUTPUT_SIZE,
        )
    elif model_type == "lstm":
        return LSTMAssociation(
            input_size=LSTM_INPUT_SIZE,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            max_dets=MAX_DETS,
            forget_bias=LSTM_FORGET_BIAS,
        )
    else:
        raise ValueError("model_type must be rnn or lstm")


def validate_rnn(model, val_loader, criterion, device):
    """Validate RNN motion model."""
    model.eval()
    total_loss = 0.0
    details = {}
    n_batches = 0
    with torch.no_grad():
        for (
            bbox_in,
            bbox_tgt,
            exist_in,
            exist_tgt,
        ) in val_loader:
            bbox_in = bbox_in.to(device)
            bbox_tgt = bbox_tgt.to(device)
            exist_tgt = exist_tgt.to(device)
            pred, updated, exist, _, _ = model(bbox_in)
            loss, det = criterion(
                pred=pred,
                updated=updated,
                exist=exist,
                gt_bbox=bbox_tgt,
                gt_exist=exist_tgt,
            )
            total_loss += loss.item()
            details = det
            n_batches += 1
    n_batches = max(n_batches, 1)
    return total_loss / n_batches, details


def validate_lstm(model, val_loader, criterion, device):
    """Validate LSTM association model."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    with torch.no_grad():
        for (
            C_flat,
            labels,
            n_tracks,
            n_dets,
        ) in val_loader:
            C_flat = C_flat.to(device)
            labels = labels.to(device)
            n_tracks = n_tracks.to(device)
            n_dets = n_dets.to(device)
            B = C_flat.shape[0]
            all_probs = torch.zeros(
                B,
                MAX_TRACKS,
                MAX_DETS,
                device=device,
            )
            for b in range(B):
                nt = n_tracks[b].item()
                nd = int(n_dets[b].item())
                if nt == 0:
                    continue
                C_mat = C_flat[b].view(MAX_TRACKS, MAX_DETS)
                hidden = model.init_hidden(1)
                for i in range(nt):
                    row_i = C_mat[i].unsqueeze(0).unsqueeze(0)
                    probs, hidden = model(row_i, hidden)
                    raw = probs[0]
                    valid = raw[:nd]
                    p_sum = valid.sum()
                    if p_sum > 1e-8:
                        valid = valid / p_sum
                    pad = torch.zeros(
                        MAX_DETS - nd,
                        device=raw.device,
                    )
                    p = torch.cat([valid, pad], dim=0)
                    all_probs[b, i] = p
            loss, det = criterion(
                all_probs,
                labels,
                n_tracks,
                n_dets,
            )
            total_loss += det["loss"]
            total_acc += det["accuracy_pct"]
            n_batches += 1
    n_batches = max(n_batches, 1)
    return (
        total_loss / n_batches,
        total_acc / n_batches,
    )


def plot_training_curves(
    train_losses,
    val_records,
    out_dir,
    mode="rnn",
):
    """Plot training loss and validation loss curves."""
    os.makedirs(out_dir, exist_ok=True)

    title = (
        "RNN Motion Model: Train Loss vs Val Loss"
        if mode == "rnn"
        else "LSTM Association Model: Train Loss vs Val Loss"
    )

    # Different smoothing for rnn vs lstm
    window = 500 if mode == "rnn" else 100

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle(title)

    if len(train_losses) > window:
        smooth = np.convolve(
            train_losses,
            np.ones(window) / window,
            mode="valid",
        )
        ax.plot(
            smooth,
            color="blue",
            linewidth=2,
            label="Train Loss",
        )
    else:
        ax.plot(
            train_losses,
            color="blue",
            linewidth=2,
            label="Train Loss",
        )

    if val_records:
        val_iters = [v[0] for v in val_records]
        val_losses = [v[1] for v in val_records]
        ax.plot(
            val_iters,
            val_losses,
            color="orange",
            linewidth=2,
            label="Val Loss",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(
        out_dir,
        "training_curves_{}.png".format(mode),
    )
    plt.savefig(out, dpi=150)
    plt.show()
    plt.close()
    print("Curves saved: {}".format(out_dir))


def plot_sample_frame():
    """
    Display a sample CARLA frame
    with ground truth bounding boxes.
    """
    if not os.path.exists(SAMPLE_BBOX_IMAGE):
        print("Image not found: {}".format(SAMPLE_BBOX_IMAGE))
        return

    img = np.array(Image.open(SAMPLE_BBOX_IMAGE))
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.imshow(img)
    ax.set_title("Sample Frame")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_rnn_predictions(rnn_model, device, n_tracks=3):
    """
    Plot RNN predictions using best checkpoint
    against ground truth trajectories on test set.
    Args:
        rnn_model: trained RNNMotionModel
        device   : torch device
        n_tracks : number of tracks to plot
    """
    # Load best checkpoint for best predictions
    ckpt_path = os.path.join(CHECKPOINT_DIR, "rnn_real_best.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        rnn_model.load_state_dict(checkpoint["state_dict"])
        print(
            "Loaded best RNN checkpoint "
            "(val loss={:.4f})".format(checkpoint["val_loss"])
        )

    # Load test data
    test_ds = RealTrajectoryDataset(split="test", seq_length=SEQ_LENGTH)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    print("Test sequences: {:,}".format(len(test_ds)))

    rnn_model.eval()

    sample = next(iter(test_loader))
    bbox_in, bbox_tgt, _, _ = sample
    bbox_in = bbox_in[:n_tracks].to(device)
    bbox_tgt = bbox_tgt[:n_tracks].to(device)

    with torch.no_grad():
        pred, _, _, _, _ = rnn_model(bbox_in)

    bbox_tgt = bbox_tgt.cpu().numpy()
    pred = pred.cpu().numpy()

    fig, axes = plt.subplots(1, n_tracks, figsize=(15, 4))
    if n_tracks == 1:
        axes = [axes]

    for i in range(n_tracks):
        axes[i].plot(
            bbox_tgt[i, :, 0],
            bbox_tgt[i, :, 1],
            "g-o",
            label="Ground Truth",
            markersize=4,
        )
        axes[i].plot(
            pred[i, :, 0],
            pred[i, :, 1],
            "r--s",
            label="RNN Prediction",
            markersize=4,
        )
        axes[i].set_title("Track {}".format(i + 1))
        axes[i].set_xlabel("x (normalised)")
        axes[i].set_ylabel("y (normalised)")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(
        "RNN Motion Model: "
        "Predicted vs Ground Truth Trajectories"
        " (Test Set - Town07)"
    )
    plt.tight_layout()
    plt.show()


def test_and_plot_lstm(lstm_model, lstm_val_loader, device):
    """
    Test LSTM model and visualise output
    assignment probabilities as single matrix.
    Args:
        lstm_model     : trained LSTMAssociation
        lstm_val_loader: LSTM validation DataLoader
        device         : torch device
    """
    # Load best checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, "lstm_{}_best.pth".format(LSTM_MODE))
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        lstm_model.load_state_dict(checkpoint["state_dict"])
        print(
            "Loaded best LSTM checkpoint "
            "(val loss={:.4f})".format(checkpoint["val_loss"])
        )

    lstm_model.eval()

    # Get sample internally
    sample = next(iter(lstm_val_loader))
    C_flat, labels, n_tracks, n_dets = sample

    n_t = n_tracks[0].item()
    n_d = int(n_dets[0].item())
    C_mat = C_flat[0].view(MAX_TRACKS, MAX_DETS)
    labels_s = labels[0, :n_t].numpy()

    with torch.no_grad():
        hidden = lstm_model.init_hidden(1)
        all_probs = []
        for i in range(n_t):
            row_i = C_mat[i].unsqueeze(0).unsqueeze(0).to(device)
            probs, hidden = lstm_model(row_i, hidden)
            valid = probs[0, :n_d].cpu()
            p_sum = valid.sum()
            if p_sum > 1e-8:
                valid = valid / p_sum
            all_probs.append(valid.numpy())

    prob_matrix = np.array(all_probs)

    # Tests
    assert prob_matrix.shape == (n_t, n_d), "LSTM output shape incorrect!"
    print("LSTM model test passed.")
    print("  output shape: {}".format(prob_matrix.shape))
    print("  probs sum   : {:.6f}".format(prob_matrix[0].sum()))

    # Single matrix plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    im = ax.imshow(
        prob_matrix,
        cmap="gray_r",
        aspect="auto",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(im, ax=ax, label="Assignment probability (0=low, 1=high)")

    # Mark ground truth (red box)
    for i in range(n_t):
        gt = labels_s[i]
        if 0 <= gt < n_d:
            ax.add_patch(
                plt.Rectangle(
                    (gt - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                    label="Ground truth" if i == 0 else "",
                )
            )

    # Mark LSTM prediction (blue dashed box)
    for i in range(n_t):
        pred_det = prob_matrix[i].argmax()
        ax.add_patch(
            plt.Rectangle(
                (pred_det - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="blue",
                linewidth=2,
                linestyle="--",
                label="LSTM prediction" if i == 0 else "",
            )
        )

    ax.set_title(
        "LSTM Association Model Output\n"
        "Assignment Probabilities\n"
        "(red = ground truth, "
        "blue dashed = LSTM prediction)"
    )
    ax.set_xlabel("Detection index")
    ax.set_ylabel("Track index")
    ax.set_xticks(range(n_d))
    ax.set_yticks(range(n_t))
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


def export_models(
    rnn_model,
    lstm_model,
    val_loader,
    lstm_val_loader,
    cs,
):
    """
    Export RNN and LSTM models to ONNX format.
    Uses real CARLA data as sample input.
    Args:
        rnn_model      : trained RNNMotionModel
        lstm_model     : trained LSTMAssociation
        val_loader     : RNN validation DataLoader
        lstm_val_loader: LSTM validation DataLoader
        cs             : CentralScripts helper
    Returns:
        model_names: list of exported model names
    """
    import logging
    import sys

    logging.getLogger("torch.onnx").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")

    origin = "torch"
    model_names = [
        "vehicle_tracker_rnn",
        "vehicle_tracker_lstm",
    ]

    try:
        # RNN export
        rnn_sample = next(iter(val_loader))
        bbox_in, _, _, _ = rnn_sample
        input_rnn = bbox_in[0, 0, :].numpy().reshape(1, 4).astype(np.float32)

        rnn_export = RNNPredictionOnly(rnn_model)
        rnn_export.eval()
        output_rnn = cs.get_predictions(origin, rnn_export, input_rnn)
        cs.save_all(model_names[0], input_rnn, output_rnn, rnn_export, origin)
        cs.test_onnx_pb(model_names[0])
        print("RNN exported.")

        # LSTM export
        lstm_sample = next(iter(lstm_val_loader))
        C_flat, _, _, _ = lstm_sample
        C_mat = C_flat[0].view(MAX_TRACKS, MAX_DETS)
        input_lstm = C_mat[0].numpy().reshape(1, MAX_DETS).astype(np.float32)

        lstm_export = LSTMProbsOnly(lstm_model)
        lstm_export.eval()
        output_lstm = cs.get_predictions(origin, lstm_export, input_lstm)
        cs.save_all(model_names[1], input_lstm, output_lstm, lstm_export, origin)
        cs.test_onnx_pb(model_names[1])
        print("LSTM exported.")
        print("All exports complete.")

    finally:
        sys.stderr.close()
        sys.stderr = stderr
        warnings.filterwarnings("default")
        logging.getLogger("torch.onnx").setLevel(logging.WARNING)

    return model_names


def train_rnn(
    train_loader,
    val_loader,
    device,
    checkpoint_dir,
    max_iters=MAX_ITERS,
    val_freq=1000,
):
    """
    Train RNN motion model on CARLA trajectories.
    Args:
        train_loader  : DataLoader for training
        val_loader    : DataLoader for validation
        device        : torch device
        checkpoint_dir: path to save checkpoints
        max_iters     : number of training iterations
        val_freq      : validation frequency
    Returns:
        model          : trained RNNMotionModel
        train_losses   : list of training losses
        val_checkpoints: list of (iter, loss, details)
    """
    model = get_model("rnn").to(device)
    criterion = TrackingLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    iter_count = 0
    train_losses = []
    val_checkpoints = []
    t_start = time.time()
    print("Training RNN...")
    while iter_count < max_iters:
        for (
            bbox_in,
            bbox_tgt,
            exist_in,
            exist_tgt,
        ) in train_loader:
            if iter_count >= max_iters:
                break
            model.train()
            bbox_in = bbox_in.to(device)
            bbox_tgt = bbox_tgt.to(device)
            exist_tgt = exist_tgt.to(device)
            optimizer.zero_grad()
            pred, updated, exist, _, _ = model(bbox_in)
            loss, details = criterion(
                pred=pred,
                updated=updated,
                exist=exist,
                gt_bbox=bbox_tgt,
                gt_exist=exist_tgt,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            iter_count += 1
            train_losses.append(loss.item())
            if iter_count % LR_DECAY_STEP == 0:
                for pg in optimizer.param_groups:
                    pg["lr"] *= 1.0 - LR_DECAY
            if iter_count % val_freq == 0:
                val_loss, val_det = validate_rnn(
                    model,
                    val_loader,
                    criterion,
                    device,
                )
                val_checkpoints.append((iter_count, val_loss, val_det))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            "iter": iter_count,
                            "state_dict": model.state_dict(),
                            "val_loss": val_loss,
                        },
                        os.path.join(
                            checkpoint_dir,
                            "rnn_real_best.pth",
                        ),
                    )
                elapsed = time.time() - t_start
                print(
                    "Iter {:>7,} | "
                    "val loss={:.4f} | "
                    "best val loss={:.4f} | "
                    "{:.0f}s".format(
                        iter_count,
                        val_loss,
                        best_val_loss,
                        elapsed,
                    )
                )
    print("RNN training done. Best val loss: {:.4f}".format(best_val_loss))
    return model, train_losses, val_checkpoints


def train_lstm(
    train_loader,
    val_loader,
    device,
    checkpoint_dir,
    max_iters=LSTM_MAX_ITERS,
    val_freq=1000,
):
    """
    Train LSTM association model on CARLA data.
    Args:
        train_loader  : DataLoader for training
        val_loader    : DataLoader for validation
        device        : torch device
        checkpoint_dir: path to save checkpoints
        max_iters     : number of training iterations
        val_freq      : validation frequency
    Returns:
        model        : trained LSTMAssociation
        train_losses : list of training losses
        val_records  : list of (iter, loss, accuracy)
    """
    model = get_model("lstm").to(device)
    criterion = LSTMAssociationLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LSTM_LEARNING_RATE)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    iter_count = 0
    train_losses = []
    val_records = []
    t_start = time.time()
    print("Training LSTM...")
    while iter_count < max_iters:
        for (
            C_flat,
            labels,
            n_tracks,
            n_dets,
        ) in train_loader:
            if iter_count >= max_iters:
                break
            model.train()
            C_flat = C_flat.to(device)
            labels = labels.to(device)
            n_tracks = n_tracks.to(device)
            n_dets = n_dets.to(device)
            B = C_flat.shape[0]
            optimizer.zero_grad()
            all_probs_list = []
            for b in range(B):
                nt = n_tracks[b].item()
                nd = int(n_dets[b].item())
                C_mat = C_flat[b].view(MAX_TRACKS, MAX_DETS)
                hidden = model.init_hidden(1)
                track_probs = []
                for i in range(MAX_TRACKS):
                    if i < nt:
                        row_i = C_mat[i].unsqueeze(0).unsqueeze(0)
                        probs, hidden = model(row_i, hidden)
                        raw = probs[0]
                        valid = raw[:nd]
                        p_sum = valid.sum()
                        if p_sum > 1e-8:
                            valid = valid / p_sum
                        pad = torch.zeros(
                            MAX_DETS - nd,
                            device=raw.device,
                        )
                        p = torch.cat([valid, pad], dim=0)
                    else:
                        p = torch.zeros(MAX_DETS, device=device)
                    track_probs.append(p)
                all_probs_list.append(torch.stack(track_probs, dim=0))
            all_probs = torch.stack(all_probs_list, dim=0)
            loss, details = criterion(
                all_probs,
                labels,
                n_tracks,
                n_dets,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            iter_count += 1
            train_losses.append(details["loss"])
            if iter_count % LSTM_LR_DECAY_STEP == 0:
                for pg in optimizer.param_groups:
                    pg["lr"] *= 1.0 - LSTM_LR_DECAY
            if iter_count % val_freq == 0:
                val_loss, val_acc = validate_lstm(
                    model,
                    val_loader,
                    criterion,
                    device,
                )
                val_records.append((iter_count, val_loss, val_acc))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            "iter": iter_count,
                            "state_dict": model.state_dict(),
                            "val_loss": val_loss,
                            "lstm_mode": LSTM_MODE,
                        },
                        os.path.join(
                            checkpoint_dir,
                            "lstm_{}_best.pth".format(LSTM_MODE),
                        ),
                    )
                elapsed = time.time() - t_start
                print(
                    "Iter {:>7,} | "
                    "val loss={:.4f} | "
                    "val accuracy={:.1f}% | "
                    "best val loss={:.4f} | "
                    "{:.0f}s".format(
                        iter_count,
                        val_loss,
                        val_acc,
                        best_val_loss,
                        elapsed,
                    )
                )
    print("LSTM training done. Best val loss: {:.4f}".format(best_val_loss))
    return model, train_losses, val_records

import functools
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import  Dataset
from torch import flatten
from typing import Optional
import torchaudio.functional as F
import random

import sys 
sys.path.append("audioDeepFake")
from audioDeepFake.models import ShallowCNN, SimpleLSTM, DTDNN

def find_wav_files(path_to_dir: Union[Path, str]):
    paths = list(sorted(Path(path_to_dir).glob("**/*.wav")))

    if len(paths) == 0:
        return None

    return paths


def set_seed_all(seed: int = 0):

    if not isinstance(seed, int):
        seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    os.environ["PYTHONHASHSEED"] = str(seed)
    return None

SOX_SILENCE = [
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]
class AudioDataset(Dataset):

    def __init__(
        self,
        directory_or_path_list: Union[Union[str, Path], List[Union[str, Path]]],
        sample_rate: int = 16_000,
        amount: Optional[int] = None,
        normalize: bool = True,
        trim: bool = True
    ) :
        super().__init__()

        self.trim = trim
        self.sample_rate = sample_rate
        self.normalize = normalize

        if isinstance(directory_or_path_list, list):
            paths = directory_or_path_list
        elif isinstance(directory_or_path_list, Path) or isinstance(
            directory_or_path_list, str
        ):
            directory = Path(directory_or_path_list)

            paths = find_wav_files(directory)

        if amount is not None:
            paths = paths[:amount]

        self._paths = paths

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path = self._paths[index]

        waveform, sample_rate = torchaudio.load(path, normalize=self.normalize)

        if sample_rate != self.sample_rate:
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(
                path, [["rate", f"{self.sample_rate}"]], normalize=self.normalize
            )

        if self.trim:
            (
                waveform_trimmed,
                sample_rate_trimmed,
            ) = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate, SOX_SILENCE
            )

            if waveform_trimmed.size()[1] > 0:
                waveform = waveform_trimmed
                sample_rate = sample_rate_trimmed

        audio_path = str(path)

        return waveform, sample_rate, str(audio_path)

    def __len__(self) -> int:
        return len(self._paths)


class PadDataset(Dataset):
    def __init__(self, dataset: Dataset, cut: int = 64600, label=None):
        self.dataset = dataset
        self.cut = cut
        self.label = label

    def __getitem__(self, index):
        waveform, sample_rate, audio_path = self.dataset[index]
        waveform = waveform.squeeze(0)
        waveform_len = waveform.shape[0]
        if waveform_len >= self.cut:
            if self.label is None:
                return waveform[: self.cut], sample_rate, str(audio_path)
            else:
                return waveform[: self.cut], sample_rate, str(audio_path), self.label
        # need to pad
        num_repeats = int(self.cut / waveform_len) + 1
        padded_waveform = torch.tile(waveform, (1, num_repeats))[:, : self.cut][0]

        if self.label is None:
            return padded_waveform, sample_rate, str(audio_path)
        else:
            return padded_waveform, sample_rate, str(audio_path), self.label

    def __len__(self):
        return len(self.dataset)


class TransformDataset(Dataset):

    def __init__(
        self,
        dataset: Dataset,
        transformation: Callable,
        needs_sample_rate: bool = False,
        transform_kwargs: dict = {},
    ) -> None:
        super().__init__()
        self._dataset = dataset

        self._transform_constructor = transformation
        self._needs_sample_rate = needs_sample_rate
        self._transform_kwargs = transform_kwargs

        self._transform = None

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        waveform, sample_rate, audio_path = self._dataset[index]

        if self._transform is None:
            if self._needs_sample_rate:
                self._transform = self._transform_constructor(
                    sample_rate, **self._transform_kwargs
                )
            else:
                self._transform = self._transform_constructor(**self._transform_kwargs)

        return self._transform(waveform), sample_rate, str(audio_path)


class DoubleDeltaTransform(torch.nn.Module):

    def __init__(self, win_length: int = 5, mode: str = "replicate"):
        super().__init__()
        self.win_length = win_length
        self.mode = mode

        self._delta = torchaudio.transforms.ComputeDeltas(
            win_length=self.win_length, mode=self.mode
        )

    def forward(self, X):

        delta = self._delta(X)
        double_delta = self._delta(delta)

        return torch.hstack((X, delta, double_delta))


def _build_preprocessing(
    directory_or_audiodataset: Union[Union[str, Path], AudioDataset],
    transform: torch.nn.Module,
    audiokwargs: dict = {},
    transformkwargs: dict = {},
):
    if isinstance(directory_or_audiodataset, AudioDataset) or isinstance(
        directory_or_audiodataset, PadDataset
    ):
        return TransformDataset(
            dataset=directory_or_audiodataset,
            transformation=transform,
            needs_sample_rate=True,
            transform_kwargs=transformkwargs,
        )
    elif isinstance(directory_or_audiodataset, str) or isinstance(
        directory_or_audiodataset, Path
    ):
        return TransformDataset(
            dataset=AudioDataset(directory=directory_or_audiodataset, **audiokwargs),
            transformation=transform,
            needs_sample_rate=True,
            transform_kwargs=transformkwargs,
        )


mfcc = functools.partial(_build_preprocessing, transform=torchaudio.transforms.MFCC)

def double_delta(dataset: Dataset, delta_kwargs: dict = {}) -> TransformDataset:
    return TransformDataset(
        dataset=dataset,
        transformation=DoubleDeltaTransform,
        transform_kwargs=delta_kwargs,
    )


def pred_audio(audio_path):
    audio = [audio_path]
    
    audio_ds = AudioDataset(audio)
    audio_ds = PadDataset(audio_ds)
    
    audio_ds = mfcc(
        directory_or_audiodataset=audio_ds,
        transformkwargs={}
    )
    
    audio_ds = double_delta(audio_ds)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cnn_model = ShallowCNN(in_features= 1,out_dim=1).to(device)
    cnn_checkpoint = torch.load("models/best_cnn.pt", map_location=device)
    cnn_model.load_state_dict(cnn_checkpoint['state_dict'])
    
    lstm_model = SimpleLSTM(feat_dim= 40, time_dim= 972, mid_dim= 30, out_dim= 1).to(device)
    lstm_checkpoint = torch.load("models/best_lstm.pt", map_location=device)
    lstm_model.load_state_dict(lstm_checkpoint['state_dict'])
    
    
    dtdnn_model = DTDNN(feat_dim= 38880,num_classes= 1).to(device)
    dtdnn_checkpoint = torch.load("models/best_tdnn.pt", map_location=device)
    dtdnn_model.load_state_dict(dtdnn_checkpoint['state_dict'])
    
    # Set models to evaluation mode
    cnn_model.eval()
    lstm_model.eval()
    dtdnn_model.eval()
    
    # Prepare input data
    input_data = audio_ds[0][0].unsqueeze(0)
    
    # Forward pass through CNN model
    cnn_output = cnn_model(input_data)
    cnn_prob = torch.sigmoid(cnn_output)
    
    # Forward pass through LSTM model
    lstm_output = lstm_model(input_data)
    lstm_prob = torch.sigmoid(lstm_output)
    
    # Forward pass through DT-DNN model
    dtdnn_input = input_data.view(input_data.size(0), -1)
    dtdnn_output = dtdnn_model(dtdnn_input)
    dtdnn_prob = torch.sigmoid(dtdnn_output)
    
    # Combine predictions
    combined_prob = (cnn_prob + lstm_prob + dtdnn_prob) / 3
    
    # Classify based on combined probabilities
    combined_pred = (combined_prob >= 0.5).int()
    
    return combined_pred.item()

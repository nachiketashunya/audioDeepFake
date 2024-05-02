import torchaudio  # for audio processing
import torch
from pathlib import Path
from typing import Optional, Union
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import functools
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from typing import Optional
from utils import find_wav_files

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
        waveform, sample_rate, audio_path, label = self._dataset[index]

        if self._transform is None:
            if self._needs_sample_rate:
                self._transform = self._transform_constructor(
                    sample_rate, **self._transform_kwargs
                )
            else:
                self._transform = self._transform_constructor(**self._transform_kwargs)

        return self._transform(waveform), sample_rate, str(audio_path), label


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

def double_delta(dataset: Dataset, delta_kwargs: dict = {}) -> TransformDataset:
    return TransformDataset(
        dataset=dataset,
        transformation=DoubleDeltaTransform,
        transform_kwargs=delta_kwargs,
    )


def load_directory_split_train_test(
    path: Union[Path, str],
    feature_fn: Callable,
    feature_kwargs: dict,
    test_size: float,
    use_double_delta: bool = True,
    pad: bool = False,
    label: Optional[int] = None,
):
    
    paths = find_wav_files(path)

    test_size = int(test_size * len(paths))

    train_paths = paths[:-test_size]
    test_paths = paths[-test_size:]

    train_dataset = AudioDataset(train_paths)
    if pad:
        train_dataset = PadDataset(train_dataset, label=label)

    test_dataset = AudioDataset(test_paths)
    if pad:
        test_dataset = PadDataset(test_dataset, label=label)

    dataset_train = feature_fn(
        directory_or_audiodataset=train_dataset,
        transformkwargs=feature_kwargs,
    )

    dataset_test = feature_fn(
        directory_or_audiodataset=test_dataset,
        transformkwargs=feature_kwargs,
    )
    if use_double_delta:
        dataset_train = double_delta(dataset_train)
        dataset_test = double_delta(dataset_test)

    return dataset_train, dataset_test
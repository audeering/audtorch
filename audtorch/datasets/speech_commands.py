import os
import random

from .utils import (download_url, extract_archive, safe_path)
from .base import AudioDataset
from .utils import safe_path


class SpeechCommands(AudioDataset):
    """

    Google Speech Commands V2

    Args:
        root (str): root directory of data set, where the CSV files are
            located, e.g. `/data/speech_commands_v0.02`
        train (bool, optional): Return the training split. `False` returns the
            test split. Default: True
        download (bool, optional): Download the dataset to `root` if it's not
            already available. Default: False (TODO)
        sampling_rate (int, optional):
        include (list of str, optional): list of categories to include as
            commands.
            If `None` all categories are included. Default:
            `['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']`
        silence (bool, optional): include a 'silence' class composed of background noise.
            Default: `True`
        transform (callable, optional): function/transform applied on the
            signal. Default: `None`
        target_transform (callable, optional): function/transform applied on
            the target. Default: `None`
    """

    # Available target commands
    commands = [
        'right', 'eight', 'cat', 'tree', 'backward',
        'learn', 'bed', 'happy', 'go', 'dog', 'no',
        'wow', 'follow', 'nine', 'left', 'stop', 'three',
        'sheila', 'one', 'bird', 'zero', 'seven', 'up',
        'visual', 'marvin', 'two', 'house', 'down', 'six',
        'yes', 'on', 'five', 'forward', 'off', 'four']

    background_noise = '_background_noise_'

    def __init__(self, root, train=True, download=False, *, sampling_rate=16000,
                 include=None, silence=True, transform=None, target_transform=None):

        if download:
            self._download()

        if include is None:
            include = self.commands

        with open(safe_path(os.path.join(root, 'testing_list.txt'))) as f:
            test_files = f.read().splitlines()

        files, targets = [], []
        for speech_cmd in self.commands:
            d = os.listdir(os.path.join(root, speech_cmd))
            d = [os.path.join(speech_cmd, x) for x in d]

            # Filter out test / train files using `testing_list.txt`
            d_f = list(set(d) - set(test_files)) if train else list(set(d) & set(test_files))

            files.extend([os.path.join(root, p) for p in d_f])
            target = speech_cmd if speech_cmd in include else 'unknown'
            targets.extend([target for _ in range(len(d_f))])

        # Match occurrences of silence with `unknown`
        if silence:
            n_samples = max(targets.count('unknown'), 3_000)
            n_samples = int(n_samples * 0.9) if train else int(n_samples * 0.1)

            sfiles = []
            for file in os.listdir(os.path.join(root, '_background_noise_')):
                if file.endswith('.wav'):
                    sfiles.append(os.path.join(root, '_background_noise_', file))

            targets.extend(['silence' for _ in range(n_samples)])
            files.extend(random.choices(sfiles,k=n_samples))

        super().__init__(root, files, targets, sampling_rate, transform=transform, target_transform=target_transform)

    def _download(self):
        return NotImplementedError("TODO: Automatically download dataset to root dir")
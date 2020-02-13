import os
import random
from warnings import warn

import resampy
from .utils import (download_url, extract_archive, safe_path, load)
from .base import AudioDataset
from ..transforms import RandomCrop
from os.path import join

__doctest_skip__ = ['*']


class SpeechCommands(AudioDataset):
    r"""Data set of spoken words designed for keyword spotting tasks.

    Speech Commands V2 publicly available from Google:
    http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

    License: CC BY 4.0

    Args:
        root (str): root directory of data set,
            where the CSV files are located,
            e.g. `/data/speech_commands_v0.02`
        train (bool, optional): Partition the dataset into the training set.
            `False` returns the test split.
            Default: `False`
        download (bool, optional): Download the dataset to `root`
            if it's not already available.
            Default: `False`
        include (str, or list of str, optional): commands to include
            as 'recognised' words.
            Options: `"10cmd"`, `"full"`.
            A custom dataset can be defined using a list of command words.
            For example, `["stop","go"]`.
            Words that are not in the "include" list
            are treated as unknown words.
            Default: `'10cmd'`
        silence (bool, optional): include a 'silence' class composed of
            background noise (Note: use randomcrop when training).
            Default: `True`
        transform (callable, optional): function/transform applied on the
            signal.
            Default: `None`
        target_transform (callable, optional): function/transform applied on
            the target.
            Default: `None`

    Example:
        >>> import sounddevice as sd
        >>> data = SpeechCommands(root='/data/speech_commands_v0.02')
        >>> print(data)
        Dataset SpeechCommands
            Number of data points: 97524
            Root Location: /data/speech_commands_v0.02
            Sampling Rate: 16000Hz
        >>> signal, target = data[4]
        >>> target
        'right'
        >>> sd.play(signal.transpose(), data.sampling_rate)
    """

    url = ('http://download.tensorflow.org/'
           'data/speech_commands_v0.02.tar.gz')

    # Available target commands
    classes = [
        'right', 'eight', 'cat', 'tree', 'backward',
        'learn', 'bed', 'happy', 'go', 'dog', 'no',
        'wow', 'follow', 'nine', 'left', 'stop', 'three',
        'sheila', 'one', 'bird', 'zero', 'seven', 'up',
        'visual', 'marvin', 'two', 'house', 'down', 'six',
        'yes', 'on', 'five', 'forward', 'off', 'four']

    partitions = {
        # https://arxiv.org/pdf/1710.06554.pdf
        '10cmd': ['yes', 'no', 'up', 'down', 'left',
                  'right', 'on', 'off', 'stop', 'go'],
        'full': classes
    }

    def __init__(
            self,
            root,
            train=True,
            download=False,
            *,
            sampling_rate=16000,
            include='10cmd',
            transform=None,
            target_transform=None,
    ):
        self.root = safe_path(root)
        self.same_length = False
        self.silence_label = -1
        self.trim = RandomCrop(sampling_rate)

        if download:
            self._download()

        if type(include) is not list:
            include = self.partitions[include]

        if not set(include) == set(self.classes):
            include.append("_unknown_")

        with open(safe_path(join(self.root, 'testing_list.txt'))) as f:
            test_files = f.read().splitlines()

        files, targets = [], []
        for speech_cmd in self.classes:
            d = os.listdir(join(self.root, speech_cmd))
            d = [join(speech_cmd, x) for x in d]

            # Filter out test / train files using `testing_list.txt`
            d_f = list(set(d) - set(test_files)) \
                if train else list(set(d) & set(test_files))

            files.extend([join(self.root, p) for p in d_f])
            target = speech_cmd if speech_cmd in include else '_unknown_'
            # speech commands is a classification dataset, so return logits
            targets.extend([include.index(target) for _ in range(len(d_f))])

        self.silence_label = len(include)

        # Match occurrences of silence with `unknown`
        # if silence:
        #     n_samples = max(targets.count(len(include) - 1), 3000)
        #     n_samples = int(n_samples * 0.9) \
        #         if train else int(n_samples * 0.1)
        #
        #     sf = []
        #     for file in os.listdir(join(self.root, '_background_noise_')):
        #         if file.endswith('.wav'):
        #             sf.append(join(self.root, '_background_noise_', file))
        #
        #     targets.extend([len(include) for _ in range(n_samples)])
        #     files.extend(random.choices(sf, k=n_samples))

        super().__init__(
            files=files,
            targets=targets,
            sampling_rate=sampling_rate,
            root=root,
            transform=transform,
            target_transform=target_transform,
        )

    def add_silence(
            self,
            n_samples=3000,
            same_length=True,
    ):
        # https://github.com/audeering/audtorch/pull/49#discussion_r317489141
        self.same_length = same_length
        self.targets.extend([self.silence_label for _ in range(n_samples)])

        bg_noises = []
        for file in os.listdir(join(self.root, '_background_noise_')):
            if file.endswith('.wav'):
                bg_noises.append(join(self.root, '_background_noise_', file))

        self.files.extend(random.choices(bg_noises, k=n_samples))

    def __getitem__(self, index):
        signal, signal_sampling_rate = load(self.files[index])
        # Handle empty signals
        if signal.shape[1] == 0:
            warn('Returning previous file.', UserWarning)
            return self.__getitem__(index - 1)
        # Handle different sampling rate
        if signal_sampling_rate != self.original_sampling_rate:
            warn(
                (f'Resample from {signal_sampling_rate} '
                 f'to {self.original_sampling_rate}'),
                UserWarning,
            )
            signal = resampy.resample(signal, signal_sampling_rate,
                                      self.original_sampling_rate, axis=-1)

        target = self.targets[index]

        # https://github.com/audeering/audtorch/pull/49#discussion_r319044362
        if target == self.silence_label and self.same_length:
            signal = self.trim(signal)

        if self.transform is not None:
            signal = self.transform(signal)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return signal, target

    def _download(self):
        if self._check_exists():
            return
        download_dir = self.root
        corpus = 'speech_commands_v0.02'
        if download_dir.endswith(corpus):
            download_dir = download_dir[:-len(corpus)]
        filename = download_url(self.url, download_dir)
        extract_archive(filename, out_path=self.root, remove_finished=True)

import os
import random

from .utils import (download_url, extract_archive, safe_path)
from .base import AudioDataset
from os.path import join

__doctest_skip__ = ['*']


class SpeechCommands(AudioDataset):
    r"""Data set of spoken words designed for keyword spotting tasks.

    Speech Commands V2 publicly available from Google:
    http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

    License: CC BY 4.0

    Args:
        root (str): root directory of data set, where the CSV files are
            located, e.g. `/data/speech_commands_v0.02`
        train (bool, optional): Partition the dataset into the training set.
            `False` returns the test split. Default: False.
        download (bool, optional): Download the dataset to `root` if it's not
            already available. Default: False
        include (str, or list of str): list of commands to include as recognised
            words. options: `"10cmd"`, `"full"`. A custom dataset can be defined
            using a list of command words. For example, ["stop","go"]. Words that
            are not in the "include" list are treated as unknown words.
        silence (bool, optional): include a 'silence' class composed of
            background noise. (Note: use randomcrop when training)
            Default: `True`
        transform (callable, optional): function/transform applied on the
            signal. Default: `None`
        target_transform (callable, optional): function/transform applied on
            the target. Default: `None`

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
    commands = [
        'right', 'eight', 'cat', 'tree', 'backward',
        'learn', 'bed', 'happy', 'go', 'dog', 'no',
        'wow', 'follow', 'nine', 'left', 'stop', 'three',
        'sheila', 'one', 'bird', 'zero', 'seven', 'up',
        'visual', 'marvin', 'two', 'house', 'down', 'six',
        'yes', 'on', 'five', 'forward', 'off', 'four']

    background_noise = '_background_noise_'

    partitions = {
        # https://arxiv.org/pdf/1710.06554.pdf
        '10cmd': ['yes', 'no', 'up', 'down', 'left',
                  'right', 'on', 'off', 'stop', 'go',
                  '_silence_', '_unknown_'],
        'full':  list(commands).extend(['_silence_'])
    }

    def __init__(self, root, train=True, download=False, *,
                 sampling_rate=16000, include='10cmd', silence=True,
                 transform=None, target_transform=None):
        self.root = safe_path(root)

        if download:
            self._download()

        if type(include) is not list:
            include = self.partitions[include]

        with open(safe_path(join(self.root, 'testing_list.txt'))) as f:
            test_files = f.read().splitlines()

        files, targets = [], []
        for speech_cmd in self.commands:
            d = os.listdir(join(self.root, speech_cmd))
            d = [join(speech_cmd, x) for x in d]

            # Filter out test / train files using `testing_list.txt`
            d_f = list(set(d) - set(test_files)) \
                if train else list(set(d) & set(test_files))

            files.extend([join(self.root, p) for p in d_f])
            target = speech_cmd if speech_cmd in include else '_unknown_'
            targets.extend([target for _ in range(len(d_f))])

        # Match occurrences of silence with `unknown`
        if silence:
            n_samples = max(targets.count('_unknown_'), 3000)
            n_samples = int(n_samples * 0.9) \
                if train else int(n_samples * 0.1)

            sf = []
            for file in os.listdir(join(self.root, '_background_noise_')):
                if file.endswith('.wav'):
                    sf.append(join(self.root, '_background_noise_', file))

            targets.extend(['_silence_' for _ in range(n_samples)])
            files.extend(random.choices(sf, k=n_samples))

        super().__init__(root, files, targets, sampling_rate,
                         transform=transform,
                         target_transform=target_transform)

    def _download(self):
        if self._check_exists():
            return
        download_dir = self.root
        corpus = 'speech_commands_v0.02'
        if download_dir.endswith(corpus):
            download_dir = download_dir[:-len(corpus)]
        filename = download_url(self.url, download_dir)
        extract_archive(filename, out_path=self.root, remove_finished=True)

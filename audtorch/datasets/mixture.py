import random

import numpy as np
from torch.utils.data import Dataset

from .base import _include_repr
from .utils import ensure_same_sampling_rate


__doctest_skip__ = ['*']


class SpeechNoiseMix(Dataset):
    r"""Mix speech and noise with speech as target.

    Add noise to each speech sample from the provided data by a
    mix transform. Return the mix as input and the speech signal as
    corresponding target. In addition, allow to replace randomly some of the
    mixes by noise as input and silence as output. This helps to train a speech
    enhancement algorithm to deal with background noise only as input signal
    :cite:`Rethage2018`.

    * :attr:`speech_dataset` controls the speech data set
    * :attr:`mix_transform` controls the transform that adds noise
    * :attr:`transform` controls the transform applied on the mix
    * :attr:`target_transform` controls the transform applied on the target
      clean speech
    * :attr:`joint_transform` controls the transform applied jointly on the
      mixture an the target clean speech
    * :attr:`percentage_silence` controls the amount of noise-silent data
      augmentation

    Args:
        speech_dataset (Dataset): speech data set
        mix_transform (callable): function/transform that can augment a signal
            with noise
        transform (callable, optional): function/transform applied on the
            speech-noise-mixture (input) only. Default; `None`
        target_transform (callable, optional): function/transform applied
            on the speech (target) only. Default: `None`
        joint_transform (callable, optional): function/transform applied
            on the mixtue (input) and speech (target) simultaneously. If the
            transform includes randomization it is applied with the same random
            parameter during both calls
        percentage_silence (float, optional): value between `0` and `1`, which
            controls the percentage of randomly inserted noise input, silent
            target pairs. Default: `0`

    Examples:
        >>> import sounddevice as sd
        >>> from audtorch import datasets, transforms
        >>> noise = datasets.WhiteNoise(duration=10, sampling_rate=48000)
        >>> mix = transforms.RandomAdditiveMix(noise)
        >>> normalize = transforms.Normalize()
        >>> speech = datasets.MozillaCommonVoice(root='/data/MozillaCommonVoice/cv_corpus_v1')
        >>> data = SpeechNoiseMix(speech, mix, transform=normalize)
        >>> noisy, clean = data[0]
        >>> sd.play(noisy.transpose(), data.sampling_rate)

    """  # noqa: E501
    def __init__(
            self,
            speech_dataset,
            mix_transform,
            *,
            transform=None,
            target_transform=None,
            joint_transform=None,
            percentage_silence=0,
    ):
        super().__init__()
        self.speech_dataset = speech_dataset
        self.mix_transform = mix_transform
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.percentage_silence = percentage_silence

        if not (0 <= self.percentage_silence <= 1):
            raise ValueError('`percentage_silence` needs to be in [0, 1]`')

        if hasattr(mix_transform, 'dataset'):
            ensure_same_sampling_rate([speech_dataset, mix_transform.dataset])

    def __len__(self):
        return len(self.speech_dataset)

    def __getitem__(self, item):
        # [0] ensures that we get only data, no targets
        speech = self.speech_dataset[item][0]
        # Randomly add (noise, silence) as (input, target)
        if random.random() < self.percentage_silence:
            speech = np.zeros(speech.shape)
        mixture = self.mix_transform(speech)

        if self.joint_transform is not None:
            randomness = getattr(self.joint_transform, 'fix_randomization',
                                 None)
            mixture = self.joint_transform(mixture)
            if randomness is not None:
                self.joint_transform.fix_randomization = True
            speech = self.joint_transform(speech)
            if randomness is not None:
                self.joint_transform.fix_randomization = randomness

        if self.transform is not None:
            mixture = self.transform(mixture)

        if self.target_transform is not None:
            speech = self.target_transform(speech)

        # input, target
        return mixture, speech

    @property
    def sampling_rate(self):
        return self.speech_dataset.sampling_rate

    def __repr__(self):
        speech_dataset_name = self.speech_dataset.__class__.__name__
        fmt_str = f'Dataset {self.__class__.__name__}\n'
        fmt_str += f'    Number of data points: {self.__len__()}\n'
        fmt_str += f'    Speech dataset: {speech_dataset_name}\n'
        fmt_str += f'    Sampling rate: {self.sampling_rate}Hz\n'
        if self.percentage_silence > 0:
            fmt_str += (
                f'    Silence augmentation: '
                f'{100 * self.percentage_silence:.0f}%\n'
            )
        fmt_str += '    Labels: speech signal\n'
        fmt_str += _include_repr('Mixing Transform', self.mix_transform)
        if self.transform:
            fmt_str += _include_repr('Transform', self.transform)
        if self.target_transform:
            fmt_str += _include_repr('Target Transform', self.target_transform)
        if self.joint_transform:
            fmt_str += _include_repr('Joint Transform', self.joint_transform)
        return fmt_str

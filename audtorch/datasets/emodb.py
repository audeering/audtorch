import os
import glob

from typing import Callable

from audtorch.datasets.base import AudioDataset
from audtorch.datasets.utils import download_url


__doctest_skip__ = ['*']


class EmoDB(AudioDataset):
    r"""EmoDB data set.

    Open and publicly available data set of acted emotions:
    http://www.emodb.bilderbar.info/navi.html

    EmoDB is a small audio data set collected in an anechoic chamber in the
    Technical University of Berlin, it contains 5 male and 5 female speakers,
    consists of 10 unique sentences, and is annotated for 6 emotions plus a
    neutral state. The spoken language is German.

    Args:
        root: root directory of dataset
        transform: function/transform applied on the signal
        target_transform: function/transform applied on the target

    Note:
        * When using the EmoDB data set in your research, please cite
          the following publication: :cite:`burkhardt2005database`.

    Example:
        >>> import sounddevice as sd
        >>> data = EmoDB('/data/emodb')
        >>> print(data)
        Dataset EmoDB
            Number of data points: 465
            Root Location: /data/emodb
            Sampling Rate: 16000Hz
            Labels: emotion
        >>> signal, target = data[0]
        >>> target
        'A'
        >>> sd.play(signal.transpose(), data.sampling_rate)

    """
    url = ('http://www.emodb.bilderbar.info/navi.html')

    def __init__(self, root: str, *, transform: Callable = None,
                 target_transform: Callable = None):
        super().__init__(root, files=[], targets=[],
                         transform=transform,
                         sampling_rate=16000,
                         target_transform=target_transform)
        self.files = glob.glob(self.root + '/*.wav')
        self.targets = [os.path.basename(x).split('.')[0][-2]
                        for x in self.files]

    def extra_repr(self):
        fmt_str = '    Labels: emotion\n'
        return fmt_str

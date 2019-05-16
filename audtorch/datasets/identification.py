import os

import pandas as pd

from audtorch.datasets.common import AudioDataset


class VoxCeleb1(AudioDataset):
    r"""VoxCeleb1 data set.

    VoxCeleb1 is a large audio-visual data set consisting of short clips of
    human speech extracted from YouTube interviews with celebrities. It is
    free for commercial and research purposes. This data set is used to iterate
    over the audio part of the data set as provided by the official owners.

    Licence: CC BY-SA 4.0

    * :attr:`transform` controls the input transform
    * :attr:`target_transform` controls the target transform
    * :attr:`files` controls the audio files of the data set
    * :attr:`targets` controls the corresponding targets
    * :attr:`sampling_rate` holds the sampling rate of data set
    * :attr:`identification_file` holds the name of the official identification
      split of the data set

    In addition, the following class attributes are available:

    * :attr:`url` holds its URL

    Args:
        root (str): root directory of dataset
        partition (str, optional): name of a a data partition to use. No
            absolute path is possible. You are most probably interested in
            `train`, `dev`, `test` or `None`. If `None` is given, then the
            whole data set will be returned. Default: `train`.
        identification_file (str, optional): name of the file containing the
            official identification split of the data set. Default:
            `identification_file.txt`
        sampling_rate (int, optional): assumed sampling rate of the data set.
            Default: `16000`
        transform (callable, optional): function/transform applied on the
            signal. Default: `None`
        target_transform (callable, optional): function/transform applied on
            the target. Default: `None`

    Note:
        * This data set will work only if the identification file is downloaded
          as is from the official homepage. Please open it in your browser and
          copy paste its contents in a file in your computer.
        * It seems like the option to automatically download the data set
          has been removed. Now the user needs to ask for a password to
          download the data set themselves. You can find that option here:
          http://www.robots.ox.ac.uk/~vgg/data/voxceleb/.

        * When using the VoxCeleb1 data set in your research, please cite
          the following publication: :cite:`nagrani2017voxceleb`.

    Example:
        >>> import sounddevice as sd
        >>> data = VoxCeleb1('/data/voxceleb1')
        >>> print(data)
        Dataset VoxCeleb1
            Number of data points: 138361
            Root Location: /data/voxceleb1
            Sampling Rate: 16000Hz
            Labels: speaker ID
        >>> sig, target = data[0]
        >>> target
        'id10003'
        >>> sd.play(sig.transpose(), data.sampling_rate)

    """
    url = ('http://www.robots.ox.ac.uk/~vgg/data/voxceleb/')
    _partitions = {'train': 1, 'dev': 2, 'test': 3}

    def __init__(self, root, *, partition='train',
                 identification_file='identification_split.txt',
                 sampling_rate=16000,
                 transform=None, target_transform=None):
        super().__init__(root, files=[], targets=[], transform=transform,
                         sampling_rate=sampling_rate,
                         target_transform=target_transform)
        filelist = pd.read_csv(
            os.path.join(self.root, identification_file),
            sep=' ', header=None)
        self.files, self.targets = self._get_files_speaker_lists(filelist)

        if partition is not None:
            # filter indices based on identification split
            indices = [index for index, x in enumerate(filelist[0])
                       if x == self._partitions[partition]]
            self.files = [self.files[index] for index in indices]
            self.targets = [self.targets[index] for index in indices]

    def _get_files_speaker_lists(self, filelist):
        r"""Extract file names and speaker IDs.

        Args:
            filelist (pandas.DataFrame): data frame containing file list
                and speakers

        Returns:
            list: files belonging to data set
            list: speaker IDs per file

        """
        files = [os.path.join(self.root, 'wav', x)
                 for x in filelist[1]]
        speakers = [x.split('/')[0] for x in filelist[1]]
        return files, speakers

    def extra_repr(self):
        fmt_str = '    Labels: speaker ID\n'
        return fmt_str

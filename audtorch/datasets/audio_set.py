import os
import json

import pandas as pd

from ..utils import flatten_list
from .base import AudioDataset
from .utils import safe_path


__doctest_skip__ = ['*']


class AudioSet(AudioDataset):
    r"""A large-scale dataset of manually annotated audio events.

    Open and publicly available data set of audio events from Google:
    https://research.google.com/audioset/

    License: CC BY 4.0

    The categories corresponding to an audio signal are returned as a list,
    starting with those included in the top hierarchy of the
    `AudioSet ontology`_, followed by those from the second hierarchy and then
    all other categories in a random order.

    The signals to be returned can be limited by excluding or including only
    certain categories. This is achieved by first including only the desired
    categories, estimating all its parent categories and then applying the
    exclusion.

    .. _AudioSet ontology: https://research.google.com/audioset/ontology/

    * :attr:`transform` controls the input transform
    * :attr:`target_transform` controls the target transform
    * :attr:`files` controls the audio files of the data set
    * :attr:`targets` controls the corresponding targets
    * :attr:`sampling_rate` holds the sampling rate of the returned data
    * :attr:`original_sampling_rate` holds the sampling rate of the audio files
      of the data set

    Args:
        root (str, optional): root directory of dataset.
            Default: `AudioSet.root`
        csv_file (str, optional): name of a CSV file located in `root`. Can be
            one of `balanced_train_segments.csv`,
            `unbalanced_train_segments.csv`, `eval_segments.csv`.
            Default: `balanced_train_segments.csv`
        include (list of str, optional): list of categories to include.
            If `None` all categories are included. Default: `None`
        exclude (list of str, optional): list of categories to exclude.
            If `None` no category is excluded. Default: `None`
        transform (callable, optional): function/transform applied on the
            signal. Default: `None`
        target_transform (callable, optional): function/transform applied on
            the target. Default: `None`

    `AudioSet ontology`_ categories of the two top hierarchies:

    .. code-block:: none

        Human sounds            Animal                   Music
        |-Human voice           |-Domestic animals, pets |-Musical instrument
        |-Whistling             |-Livestock, farm        |-Music genre
        |-Respiratory sounds    | animals, working       |-Musical concepts
        |-Human locomotion      | animals                |-Music role
        |-Digestive             \-Wild animals           \-Music mood
        |-Hands
        |-Heart sounds,         Sounds of things         Natural sounds
        | heartbeat             |-Vehicle                |-Wind
        |-Otoacoustic emission  |-Engine                 |-Thunderstorm
        \-Human group actions   |-Domestic sounds,       |-Water
                                | home sounds            \-Fire
        Source-ambiguous sounds |-Bell
        |-Generic impact sounds |-Alarm                  Channel, environment
        |-Surface contact       |-Mechanisms             and background
        |-Deformable shell      |-Tools                  |-Acoustic environment
        |-Onomatopoeia          |-Explosion              |-Noise
        |-Silence               |-Wood                   \-Sound reproduction
        \-Other sourceless      |-Glass
                                |-Liquid
                                |-Miscellaneous sources
                                \-Specific impact sounds

    Warning:
        Some of the recordings in `AudioSet` were captured with `mono` and
        others with `stereo` input. The user must be careful to handle this,
        e.g. using a transform to adjust number of channels.

    Example:
        >>> import sounddevice as sd
        >>> data = AudioSet(root='/data/AudioSet', include=['Thunderstorm'])
        >>> print(data)
        Dataset AudioSet
            Number of data points: 73
            Root Location: /data/AudioSet
            Sampling Rate: 16000Hz
            CSV file: balanced_train_segments.csv
            Included categories: ['Thunderstorm']
        >>> signal, target = data[4]
        >>> target
        ['Natural sounds', 'Thunderstorm', 'Water', 'Rain', 'Thunder']
        >>> sd.play(signal.transpose(), data.sampling_rate)

    """

    # Categories of the two top hieararchies of AudioSet
    # https://research.google.com/audioset/ontology/
    categories = {
        'Human sounds': [
            'Human voice', 'Whistling', 'Respiratory sounds',
            'Human locomotion', 'Digestive', 'Hands',
            'Heart sounds, heartbeat', 'Otoacoustic emission',
            'Human group actions'],
        'Source-ambiguous sounds': [
            'Generic impact sounds', 'Surface contact', 'Deformable shell',
            'Onomatopoeia', 'Silence', 'Other sourceless'],
        'Animal': [
            'Domestic animals, pets',
            'Livestock, farm animals, working animals', 'Wild animals'],
        'Sounds of things': [
            'Vehicle', 'Engine', 'Domestic sounds, home sounds', 'Bell',
            'Alarm', 'Mechanisms', 'Tools', 'Explosion', 'Wood', 'Glass',
            'Liquid', 'Miscellaneous sources', 'Specific impact sounds'],
        'Music': [
            'Musical instrument', 'Music genre', 'Musical concepts',
            'Music role', 'Music mood'],
        'Natural sounds': [
            'Wind', 'Thunderstorm', 'Water', 'Fire'],
        'Channel, environment and background': [
            'Acoustic environment', 'Noise', 'Sound reproduction']
    }

    def __init__(self, root, csv_file='balanced_train_segments.csv',
                 sampling_rate=16000, include=None, exclude=None,
                 transform=None, target_transform=None):
        root = safe_path(root)
        # Allow only official CSV files as no audio paths are defined otherwise
        assert csv_file in ['eval_segments.csv', 'balanced_train_segments.csv',
                            'unbalanced_train_segments.csv']
        csv_file = os.path.join(root, csv_file)

        # Load complete ontology
        with open(os.path.join(root, 'ontology.json')) as fp:
            self.ontology = json.load(fp)

        # Get the desired filenames and categories
        df = pd.read_csv(csv_file, skiprows=2, sep=', ', engine='python')
        df = self._filename_and_ids(df)
        if include is not None:
            df = self._filter_by_categories(df, include)
        df['ids'] = df['ids'].map(self._add_parent_ids)
        if exclude is not None:
            df = self._filter_by_categories(df, exclude, exclude_mode=True)
        categories = df['ids'].map(self._convert_ids_to_categories)

        audio_folder = os.path.splitext(os.path.basename(csv_file))[0]
        file_root = os.path.join(root, audio_folder)
        files = [os.path.join(file_root, f) for f in df['filename']]

        super().__init__(root, files, targets=categories, sampling_rate=16000,
                         transform=transform,
                         target_transform=target_transform)
        self.csv_file = csv_file
        self.include = include
        self.exclude = exclude

    def _filename_and_ids(self, df):
        r"""Return data frame with filenames and IDs.

        Args:
            df (pandas.DataFrame): data frame as read in from the CSV file

        Results:
            pandas.DataFrame: data frame with columns `filename` and `ids`

        """
        df.rename(columns={'positive_labels': 'ids'}, inplace=True)
        # Translate labels from "label1,label2" to [label1, label2]
        df['ids'] = [label.strip('\"').split(',') for label in df['ids']]
        # Insert filename
        df['filename'] = (df['# YTID']
                          + '_'
                          + ['{:.3f}'.format(x) for x in df['start_seconds']]
                          + '.wav')
        return df[['filename', 'ids']]

    def _add_parent_ids(self, child_ids):
        r"""Add all parent IDs to the list of given child IDs.

        Args:
            child_ids (list of str): child IDs

        Return:
            list of str: list of child and parent IDs

        """
        ids = child_ids
        for id in child_ids:
            ids += [x['id'] for x in self.ontology if id in x['child_ids']]
        # Remove duplicates
        return list(set(ids))

    def _convert_ids_to_categories(self, ids):
        r"""Convert list of ids to sorted list of categories.

        Args:
            ids (list of str): list of IDs

        Returns:
            list of str: list of sorted categories

        """
        # Convert IDs to categories
        categories = []
        for id in ids:
            categories += [x['name'] for x in self.ontology
                           if x['id'] == id]
        # Order categories after the first two top ontologies
        order = []
        first_hierarchy = self.categories.keys()
        second_hierarchy = flatten_list(list(self.categories.values()))
        for cat in categories:
            if cat in first_hierarchy:
                order += [0]
            elif cat in second_hierarchy:
                order += [1]
            else:
                order += [2]
        # Sort list `categories` by the list `order`
        categories = [cat for _, cat in sorted(zip(order, categories))]
        return categories

    def _filter_by_categories(self, df, categories, exclude_mode=False):
        r"""Return data frame containing only specified categories.

        Args:
            df (pandas.DataFrame): data frame containing the columns `ids`
            categories (list of str): list of categories to include or exclude
            exclude_mode (bool, optional): if `False` the specified categories
                should be included in the data frame, otherwise excluded.
                Default: `False`

        Returns:
            pandas.DataFrame: data frame containing only the desired categories

        """
        ids = self._ids_for_categories(categories)
        if exclude_mode:
            # Remove rows that have an intersection of actual and desired IDs
            df = df[[False if set(row['ids']) & set(ids) else True
                     for _, row in df.iterrows()]]
        else:
            # Include rows that have an intersection of actual and desired IDs
            df = df[[True if set(row['ids']) & set(ids) else False
                     for _, row in df.iterrows()]]
        df = df.reset_index(drop=True)
        return df

    def _ids_for_categories(self, categories):
        r"""All IDs and child IDs for a given set of categories.

        Args:
            categories (list of str): list of categories

        Returns:
            list: list of IDs

        """
        ids = []
        category_ids = \
            [x['id'] for x in self.ontology if x['name'] in categories]
        for category_id in category_ids:
            ids += self._subcategory_ids(category_id)
        # Remove duplicates
        return list(set(ids))

    def _subcategory_ids(self, parent_id):
        r"""Recursively identify all IDs of a given category.

        Args:
            parent_id (unicode str): ID of parent category

        Returns:
            list: list of all children IDs and the parent ID

        """
        id_list = [parent_id]
        child_ids = \
            [x['child_ids'] for x in self.ontology if x['id'] == parent_id]
        child_ids = flatten_list(child_ids)
        # Add all subcategories
        for child_id in child_ids:
            id_list += self._subcategory_ids(child_id)
        return id_list

    def extra_repr(self):
        fmt_str = '    CSV file: {}\n'.format(os.path.basename(self.csv_file))
        if self.include:
            fmt_str += '    Included categories: {}\n'.format(self.include)
        if self.exclude:
            fmt_str += '    Excluded categories: {}\n'.format(self.exclude)
        return fmt_str

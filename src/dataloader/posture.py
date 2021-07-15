"""posture dataset."""
import json
import tensorflow_datasets as tfds
import pandas as pd
from pathlib import Path

_DESCRIPTION = """
This data set contains videos for different exam actions
"""

_CITATION = """
"""


class PostureVideoConfig(tfds.core.BuilderConfig):
    """Config for StarcraftVideo dataset."""

    def __init__(self, *, data_path, resolution, batch_frame, **kwargs):
        super(PostureVideoConfig, self).__init__(
            version=tfds.core.Version("1.0.0"),
            release_notes={
                '1.0.0': 'Initial release.',
            },
            **kwargs)
        self.data_path = Path(data_path)
        self.resolution = resolution
        self.batch_frame = batch_frame


class Posture(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for posture dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        with open(str(self.builder_config.data_path / "types.json"), "r") as f:
            self.cls_types = json.load(f)
        self.labels = pd.read_csv(
            self.builder_config.data_path / "videos.csv")
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'video': tfds.features.Video(shape=(None, self.builder_config.resolution[0],
                                                    self.builder_config.resolution[1], 3), encoding_format='jpeg'),
                'frame_idx': tfds.features.Tensor(shape=(self.batch_frame)),
                'label': tfds.features.ClassLabel(num_classes=len(self.cls_types), names=self.cls_types.keys()),
            }),
            supervised_keys=('video', 'label'),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        return {
            'train': self._generate_examples(self.builder_config.data_path),
        }

    def _generate_examples(self, path):
        for f in path.glob('*.jpg'):
            yield 'key', {
                'image': f,
                'label': 0,
            }

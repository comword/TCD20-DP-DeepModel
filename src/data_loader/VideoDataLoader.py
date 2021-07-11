from src.data_loader.VideoDataSet import VideoDataset
from base.base_dataloader import BaseDataLoader

class VideoDataLoader(BaseDataLoader):
    def __init__(self, data_path, batch_size, shuffle, validation_split, num_workers, resize_to):
        self.dataset = VideoDataset(data_path)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
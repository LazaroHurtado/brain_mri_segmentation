from torch.utils.data import random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import BrainMriDataset

class BrainMriDataLoader:
    PARAMS = {
        "batch_size": 12,
        "pin_memory": True,
        "num_workers": 4
    }

    def __init__(self, transform, image_transform = None, mask_transform = None):
        dataset = BrainMriDataset(transform = transform, 
                                  image_transform = image_transform,
                                  mask_transform = mask_transform)
        self.response_cdf = dataset.response_dist()

        train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15])

        self.train_dataloader = DataLoader(train_dataset,
                                           sampler=DistributedSampler(train_dataset),
                                           **BrainMriDataLoader.PARAMS)
        self.val_dataloader = DataLoader(val_dataset,
                                         sampler=DistributedSampler(val_dataset),
                                         **BrainMriDataLoader.PARAMS)
        self.test_dataloader = DataLoader(test_dataset,
                                          sampler=DistributedSampler(test_dataset),
                                          **BrainMriDataLoader.PARAMS)
    
    def get_train_dataloader(self):
        return self.train_dataloader
    
    def get_val_dataloader(self):
        return self.val_dataloader
    
    def get_test_dataloader(self):
        return self.test_dataloader
    
    def get_class_imbalance(self):
        return self.response_cdf
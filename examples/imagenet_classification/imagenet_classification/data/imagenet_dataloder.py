import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

IMAGENET_NUM_CLASSES = 1000
IMAGENET_INPUT_SIZE = 224


def imagenet_train_loader(data_path,
                          batch_size,
                          is_distributed,
                          workers,
                          worker_init_fn,
                          mean,
                          std):
    traindir = os.path.join(data_path, 'train')
    transforms_list = [transforms.RandomResizedCrop(IMAGENET_INPUT_SIZE),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize(mean, std)]
    transforms_composed = transforms.Compose(transforms_list)
    train_dataset = datasets.ImageFolder(traindir, transforms_composed)
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=workers,
                              worker_init_fn=worker_init_fn,
                              pin_memory=True,
                              sampler=train_sampler,
                              drop_last=True)
    return train_loader


def imagenet_val_loader(data_path,
                        batch_size,
                        mean,
                        std):
    valdir = os.path.join(data_path, 'val')
    transforms_list = [transforms.Resize(IMAGENET_INPUT_SIZE + 32),
                       transforms.CenterCrop(IMAGENET_INPUT_SIZE),
                       transforms.ToTensor(),
                       transforms.Normalize(mean, std)]
    transforms_composed = transforms.Compose(transforms_list)
    val_dataset = datasets.ImageFolder(valdir, transforms_composed)
    val_loader = DataLoader(val_dataset,
                            sampler=None,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,
                            worker_init_fn=None,
                            pin_memory=True)
    return val_loader

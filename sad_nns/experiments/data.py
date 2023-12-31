from typing import Union, List

import torch
from torchvision import datasets, transforms


DATA_DIRECTORY = '../../../data/'

datasets_dict = {
    'mnist': {'dataset': datasets.MNIST,
              'num_classes': 10,
              'num_channels': 1},
    'cifar10': {'dataset': datasets.CIFAR10,
                'num_classes': 10,
                'num_channels': 3},
    'cifar100': {'dataset': datasets.CIFAR100,
                 'num_classes': 100,
                 'num_channels': 3},
}


def _get_dataset(dataset, image_size, batch_size, split=0.9, extra_transforms=None):
    if extra_transforms is None:
        extra_transforms = []

    if dataset == 'mnist':
        transform = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize((image_size, image_size), antialias=True),
            *extra_transforms
        ])
    elif dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)),
            transforms.Resize((image_size, image_size), antialias=True),
            *extra_transforms
        ])
    elif dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761)),
            transforms.Resize((image_size, image_size), antialias=True),
            *extra_transforms
        ])
    else:
        raise ValueError('Dataset {} not supported'.format(dataset))
    
    train_set = datasets_dict[dataset]['dataset'](DATA_DIRECTORY, train=True, download=True, transform=transform)
    train_size = int(split * len(train_set))
    train_set, val_set = torch.utils.data.random_split(
        train_set, lengths=[train_size, len(train_set) - train_size])

    test_set = datasets_dict[dataset]['dataset'](DATA_DIRECTORY, train=False, download=True, transform=transform)
    return train_set, val_set, test_set


class Dataset():
    def __init__(self, dataset_name, image_size, batch_size, split=0.9, extra_transforms=None):
        self.dataset_name = dataset_name
        self.num_classes = datasets_dict[self.dataset_name]['num_classes']
        self.num_channels = datasets_dict[self.dataset_name]['num_channels']

        self.extra_transforms = extra_transforms or []
        if not isinstance(self.extra_transforms, list):
            self.extra_transforms = [self.extra_transforms]

        self.image_size = image_size
        self.batch_size = batch_size
        self.split = split

        # Get the datasets and loaders
        self.datasets = _get_dataset(self.dataset_name, self.image_size, self.batch_size, 
                                         split=self.split, extra_transforms=self.extra_transforms)
        loaders = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in self.datasets]
        self.train_loader, self.val_loader, self.test_loader = loaders
        
        self.classes = self.datasets[0].dataset.classes
        
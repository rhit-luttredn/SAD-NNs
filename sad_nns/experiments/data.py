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


def get_data_loaders(dataset, image_size, batch_size, split=0.9):
    if dataset == 'mnist':
        transform = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize((image_size, image_size), antialias=True)
        ])
    elif dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)),
            transforms.Resize((image_size, image_size), antialias=True)
        ])
    elif dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761)),
            transforms.Resize((image_size, image_size), antialias=True)
        ])
    else:
        raise ValueError('Dataset {} not supported'.format(dataset))
    
    train_set = datasets_dict[dataset]['dataset'](DATA_DIRECTORY, train=True, download=True, transform=transform)
    train_size = int(split * len(train_set))
    train_set, val_set = torch.utils.data.random_split(
        train_set, lengths=[train_size, len(train_set) - train_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    test_set = datasets_dict[dataset]['dataset'](DATA_DIRECTORY, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader


class Dataset():
    def __init__(self, dataset_name, image_size, batch_size, split=0.9):
        self.dataset_name = dataset_name
        self.num_classes = datasets_dict[self.dataset_name]['num_classes']
        self.num_channels = datasets_dict[self.dataset_name]['num_channels']

        self.image_size = image_size
        self.batch_size = batch_size
        self.split = split
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            self.dataset_name, self.image_size, self.batch_size, self.split)
        
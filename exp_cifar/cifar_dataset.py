import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


def cifar10_loader(data_path='/home/alexlemo', batch_size=128):
    """
    Loads the cifar10 dataset in torch-ready format
    :param data_path:
    :param batch_size:
    :return:
    """

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_data = dset.CIFAR10(data_path, train=True, transform=train_transform, download=True)

    train_data_original = dset.CIFAR10(data_path, train=True, transform=test_transform, download=True)
    test_data = dset.CIFAR10(data_path, train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                              num_workers=2, pin_memory=True)
    train_loader_original = torch.utils.data.DataLoader(train_data_original, batch_size=batch_size,
                                                        shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader, train_loader_original
    
def cifar100_loader(data_path='/home/alexlemo', batch_size=128):
    """
    Loads the cifar100 dataset in torch-ready format
    :param data_path:
    :param batch_size:
    :return:
    """

    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_data = dset.CIFAR100(data_path, train=True, transform=train_transform, download=True)

    train_data_original = dset.CIFAR100(data_path, train=True, transform=test_transform, download=True)
    test_data = dset.CIFAR100(data_path, train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                              num_workers=2, pin_memory=True)
    train_loader_original = torch.utils.data.DataLoader(train_data_original, batch_size=batch_size,
                                                        shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader, train_loader_original


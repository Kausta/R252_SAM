import os
import torch
import torchvision
import torchvision.transforms as transforms


class Transforms:

    class CIFAR10:

        class VGG:

            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        class ResNet:

            train = transforms.Compose([
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                         std=[0.24703223, 0.24348513, 0.26158784]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])
        
        class MobileNet:
            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                         std=[0.24703223, 0.24348513, 0.26158784]),
                
            ])

    CIFAR100 = CIFAR10


def loaders(dataset, path, batch_size, num_workers, transform_name, use_test=False,
            shuffle_train=True):
    ds = getattr(torchvision.datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)
    train_set = ds(path, train=True, download=True, transform=transform.train)

    if use_test:
        print('You are going to run models on the test set. Are you sure?')
        test_set = ds(path, train=False, download=True, transform=transform.test)
    else:
        print("Using train (45000) + validation (5000)")
        train_set.data = train_set.data[:-5000]
        train_set.data = train_set.data[:-5000]

        test_set = ds(path, train=True, download=True, transform=transform.test)
        test_set.data = test_set.data[-5000:]
        test_set.data = test_set.data[-5000:]
        # import ipdb; ipdb.set_trace()
    return {    
               'train': torch.utils.data.DataLoader(
                   train_set,
                   batch_size=batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   test_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
           }, len(train_set.classes)
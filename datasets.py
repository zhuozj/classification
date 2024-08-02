import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

valid_split = 0.2
batch_size = 32
root_dir = "/home/zhuozj/Workspace/03_Dataset/melon17_full/"

"""
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomRotation(degrees=(30, 70)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
"""

# 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomVerticalFlip(p=0.5),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
valid_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])

'''
""" 加载17类水果数据集 """
dataset = datasets.ImageFolder(root_dir, transform=train_transform)
dataset_test = datasets.ImageFolder(root_dir, transform=valid_transform)

print(f"classess: {dataset.classes}")
dataset_size = len(dataset)

print(f"Total number of images: {dataset_size}")
valid_size = int(valid_split * dataset_size)

indices = torch.randperm(len(dataset)).tolist()
dataset_train = Subset(dataset, indices[:-valid_size])
dataset_valid = Subset(dataset_test, indices[-valid_size:])
'''

""" 加载30类水果数据集(已分好训练集和验证集) """
train_dir = "/home/zhuozj/Workspace/03_Dataset/fruit30_split/train/"
val_dir = "/home/zhuozj/Workspace/03_Dataset/fruit30_split/val/"
dataset_train = datasets.ImageFolder(train_dir, transform=train_transform)
dataset_valid = datasets.ImageFolder(val_dir, transform=valid_transform)
dataset = dataset_train

print(f"Total training images: {len(dataset_train)}")
print(f"Total valid images: {len(dataset_valid)}")

train_loader = DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)

valid_loader = DataLoader(
    dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

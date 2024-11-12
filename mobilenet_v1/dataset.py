import torch
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict

def get_dataloaders(batch_size=128):
    # 데이터셋 로드 및 전처리
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    #train_data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    #test_data
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # 각 클래스별로 2개씩 추출
    class_counts = defaultdict(int)
    selected_indices = []

    for idx, (data, label) in enumerate(testset):
        if class_counts[label] < 5:      # 클래스 별로 가져올 데이터 수 : 현재 10개
            selected_indices.append(idx)
            class_counts[label] += 1
        if len(selected_indices) == 50:  # 총 100개의 샘플을 선택하면 종료
            break

    # 선택된 인덱스에 해당하는 데이터만 추출
    selected_data = torch.utils.data.Subset(testset, selected_indices)
    testloader =  torch.utils.data.DataLoader(selected_data, batch_size=batch_size, shuffle=False, num_workers=2)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    return trainloader, testloader


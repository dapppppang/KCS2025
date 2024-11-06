import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# GradScaler 초기화
scaler = GradScaler()

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        # GPU로 데이터 이동
        data, target = data.to(device), target.to(device)

        # 모델의 이전 gradients 초기화
        optimizer.zero_grad()

        # float32을 사용하여 순전파 실행
        # with autocast(dtype=torch.float32):
        #   output = model(data)
        #   loss = criterion(output, target)

        output = model(data)
        loss = criterion(output, target)

        # 역전파와 최적화
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 손실 기록
        running_loss += loss.item()
        if batch_idx % 100 == 99:  # 100번째 배치마다 손실 출력
            print(f'Epoch {epoch}, Batch {batch_idx + 1}, Loss: {running_loss / 100:.4f}')
            running_loss = 0.0


def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')

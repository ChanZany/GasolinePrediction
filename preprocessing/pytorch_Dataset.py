import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class Arguments:

    def __init__(self, epochs=100):
        self.batch_size = 10
        self.test_batch_size = 1000
        self.epochs = epochs
        self.lr = 0.01
        self.momentum = 0.5  # 动量
        self.no_cuda = False  # 默认使用cuda(显卡计算)
        self.seed = 1  # 随机数种子
        self.log_interval = 5  # 日志生成间隔
        self.save_model = False  # 默认不保存模型


args = Arguments()


class GasolineDataset(Dataset):
    def __init__(self,
                 #todo 清洗后的数据路径
                 file_path="../data/train_data.csv",
                 transform=None,
                 one_hot=False):
        super(GasolineDataset, self).__init__()
        self.one_hot = one_hot
        self.file_path = file_path
        self.transform = transform
        df = pd.read_csv(file_path)
        df = df.loc[df.notnull().all(axis=1)]
        self.data = df.iloc[:, 0:-1].to_numpy(dtype=np.float)
        _labels = df.iloc[:, -1].to_numpy(dtype=np.float)
        # TODO one hot 编码
        if one_hot:
            self.labels = np.array([[i - 1] for i in _labels], dtype=np.float)
        else:
            self.labels = _labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        if self.transform is not None:
            x = self.transform(x)

        # TODO 转换为data = data.clone().detach().float().requires_grad_(True)
        #         target = target.long()
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(15, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.fc3(x)


dataset = GasolineDataset()
train_loader = DataLoader(dataset=dataset, batch_size=5, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):  # <-- now it is a distributed dataset
        # model.send(data.location)  # <-- NEW: send the model to the right location
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target.reshape(output.shape[0], output.shape[1]))
        loss.backward()
        optimizer.step()
        # model.get()  # <-- NEW: get the model back
        if batch_idx % args.log_interval == 0:
            # loss = loss.get()  # <-- NEW: get the loss back
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader) * args.batch_size,
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target.reshape(output.shape[0], output.shape[1])).item()  # sum up batch loss
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, train_loader)

    # for batch_idx, (data, target) in enumerate(train_loader):
    #     print(data)
    #     print(target)

    if (args.save_model):
        torch.save(model.state_dict(), "gasoline .ckpt")

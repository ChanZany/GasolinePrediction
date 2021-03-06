import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from torch.autograd import Variable
from pytorch_Dataset import GasolineDataset
from LossFunction import My_loss


class Arguments:

    def __init__(self, epochs=100):
        self.batch_size = 10
        self.test_batch_size = 50
        self.epochs = epochs
        self.lr = 0.0001
        self.momentum = 0.5  # 动量
        self.no_cuda = False  # 默认使用cuda(显卡计算)
        self.seed = 1  # 随机数种子
        self.log_interval = 5  # 日志生成间隔
        self.save_model = False  # 默认不保存模型


args = Arguments()

loss_history = []
class Net(nn.Module):
    def __init__(self, input_n, hidden1_n, hidden2_n, output_n):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_n, hidden1_n)
        self.fc2 = nn.Linear(hidden1_n, hidden2_n)
        self.fc3 = nn.Linear(hidden2_n, output_n)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.fc3(x)


# TODO 每次修改完数据记得更新数据的路径
dataset = GasolineDataset(file_path="../data/train3_data.csv")
train_loader = DataLoader(dataset=dataset, batch_size=5, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = My_loss()


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.mse_loss(output, target.reshape(output.shape[0], output.shape[1]))
        # TODO 重新定义损失函数
        # |y' - y|/y=0.3==> |y'-y| = 0.3y ==> |y'-y|-0.3y
        loss = criterion(output, target)
        # loss = Variable((torch.abs(output - target)/target).sum(), requires_grad=True)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader) * args.batch_size,
                       100. * batch_idx / len(train_loader), loss.item()))
            loss_history.append(loss.item())

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss  # sum up batch loss

    correct = r2_score(output.cpu().numpy(), target.view_as(output).cpu().numpy())
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    # TODO 注意这个输入20是与前面train_data的特征数(列数-1)应该是相同的
    model = Net(29, 15, 7, 1).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, train_loader)

    np.array(loss_history).tofile("./loss_historys/Solution4.loss")

    x = [i for i in range(len(loss_history))]
    y = loss_history
    plt.plot(x, y)
    plt.show()

    if (args.save_model):
        torch.save(model.state_dict(), "gasoline .ckpt")

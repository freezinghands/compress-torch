
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from sys import getsizeof as sizeof
import datetime
import time
from random import randint


# inserted to smaller dataset
tr_split_len = 60000
test_split_len = 10000
# flag = 0

tm = datetime.datetime.now()
stamp = f"{tm:%Y%m%d-%H%M%S}"
outputfilename = f"compressed-{stamp}.bin"

f = open(outputfilename, 'w')
f.close()

FatigueTH = 15
is_test_mode = 0


def save_bytestream_to_file(x, filename):
    if not is_test_mode:
        return None
    with open(outputfilename, "ab") as p:
        p.write(x.detach().numpy().tobytes())
        # for i in range(randint(0, x.size()[0] - 1)):
        # i = randint(0, x.size()[0] - 1)
        # for j in range(x.size()[1]-1):
        #     for k in range(x.size()[2]-1):
        #         for h in range(x.size()[3]-1):
        #             # p.write(x[i][j][k][h].detach().numpy().tobytes())
        #             print(f"{float(x[i][j][k][h]):10} saved to file.")

# def generate_stream(x):
#     for idx, s in enumerate(x):
#         pass



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.flag = 0
        self.fatigue = 0

    def forward(self, x):
        self.fatigue += 1
        print(f"[Forward] iteration: {self.fatigue:2}")
        # print(x[0][0][0][0].type())  # it is float32 type https://pytorch.org/docs/stable/tensors.html
        x = self.conv1(x)

        if self.flag == 0:
            print(f"size of tensor: {x.size()}")
            self.flag = 1
        if self.fatigue > FatigueTH:
            save_bytestream_to_file(x, outputfilename)
        x = F.relu(x)
        if self.fatigue > FatigueTH:
            save_bytestream_to_file(x, outputfilename)
        x = self.conv2(x)
        if self.fatigue > FatigueTH:
            save_bytestream_to_file(x, outputfilename)
        x = F.relu(x)
        if self.fatigue > FatigueTH:
            save_bytestream_to_file(x, outputfilename)
        x = F.max_pool2d(x, 2)
        if self.fatigue > FatigueTH:
            save_bytestream_to_file(x, outputfilename)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        if self.fatigue > FatigueTH:
            save_bytestream_to_file(x, outputfilename)
        x = F.relu(x)
        if self.fatigue > FatigueTH:
            save_bytestream_to_file(x, outputfilename)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    global is_test_mode
    is_test_mode = 1
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def main():
    # Training settings
    args = AttrDict()
    args.update({'batch_size': 1000})
    args.update({'test_batch_size': 100})
    args.update({'epochs': 1})
    args.update({'lr': 1.0})
    args.update({'gamma': 0.7})
    args.update({'no_cuda': False})
    args.update({'dry_run': False})
    args.update({'seed': 1})
    args.update({'log_interval': 10})
    args.update({'save_model': False})

    print(f"args: {args}")
    # print(type(args))
    # Original output
    # args: Namespace(batch_size=1000, test_batch_size=100, epochs=1, lr=1.0, gamma=0.7, no_cuda=False, dry_run=False,
    #                 seed=1, log_interval=10, save_model=False)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    transform2 = transforms.Compose([
        transforms.ToTensor()
    ])
    #dataset1 = datasets.MNIST('../data', train=True, download=True,
    #                   transform=transform1)
    #dataset2 = datasets.MNIST('../data', train=False,
    #                   transform=transform1)
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform2)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform2)

    # inserted to smaller dataset
    dataset1 = torch.utils.data.random_split(dataset1, [tr_split_len, len(dataset1) - tr_split_len])[0]
    dataset2 = torch.utils.data.random_split(dataset2, [test_split_len, len(dataset2) - test_split_len])[0]

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # print(f"testloader: {test_loader.dataset.dataset}")

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("==============================")
    print(f"elapsed time: {end - start:.3} seconds")
    # f.close()

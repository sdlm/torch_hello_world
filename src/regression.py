import argparse
import copy

import torch
import torchvision
from time import time
from torch import nn, optim, cuda
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms


PATH_TO_STORE_TRAINSET = "./data/train"
PATH_TO_STORE_TESTSET = "./data/test"

IMG_WIDTH = IMG_HEIGHT = 28

INPUT_SIZE = IMG_WIDTH * IMG_HEIGHT
NUM_CLASSES = 10

DEFAULT_EPOCHS = 10

# Feed Forward network
FIRST_HIDDEN_LAYER_SIZE = 128
SECOND_HIDDEN_LAYER_SIZE = 64


def get_data_loaders():
    train_set = torchvision.datasets.MNIST(
        PATH_TO_STORE_TRAINSET, download=True, train=True, transform=torchvision.transforms.ToTensor()
    )
    val_set = torchvision.datasets.MNIST(
        PATH_TO_STORE_TESTSET, download=True, train=False, transform=torchvision.transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True, drop_last=True)

    return train_loader, val_loader


def get_model():
    return nn.Sequential(
        nn.Linear(INPUT_SIZE, FIRST_HIDDEN_LAYER_SIZE),
        nn.ReLU(),
        nn.Linear(FIRST_HIDDEN_LAYER_SIZE, SECOND_HIDDEN_LAYER_SIZE),
        nn.ReLU(),
        nn.Linear(SECOND_HIDDEN_LAYER_SIZE, NUM_CLASSES),
    )


class FeedForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(INPUT_SIZE, FIRST_HIDDEN_LAYER_SIZE)
        self.linear2 = torch.nn.Linear(FIRST_HIDDEN_LAYER_SIZE, SECOND_HIDDEN_LAYER_SIZE)
        self.linear3 = torch.nn.Linear(SECOND_HIDDEN_LAYER_SIZE, NUM_CLASSES)

    def forward(self, x):
        out = x.reshape(x.size(0), -1)  # flat
        out = F.relu(self.linear1(out))  # .clamp(min=0)
        out = F.relu(self.linear2(out))  # .clamp(min=0)
        return self.linear3(out)


class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(kernel_size=2)  # 26 * 26  # 13 * 13
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(), nn.MaxPool2d(kernel_size=2)  # 11 * 11  # 5 * 5
        )
        self.drop_out = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(5 * 5 * 64, 1000)  # Fully Connected
        self.fc2 = nn.Linear(1000, 10)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)  # flat
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return self.softmax(out)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False


def train_model(train_loader, model, criterion, epochs):
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    time0 = time()
    # epochs = 15
    for epoch in range(epochs):
        running_loss = 0
        exp_lr_scheduler.step(epoch)
        for images, labels in train_loader:
            # Flatten MNIST images into a 784 long vector
            # images = images.view(images.shape[0], -1)
            if cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            # Training pass
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            # This is where the model learns by backpropagating
            loss.backward()

            # And optimizes its weights here
            optimizer.step()

            running_loss += loss.item()
        else:
            print(
                "Epoch{:> 3} - Train loss: {:.5f} - Elapsed: {:.2f} min".format(
                    epoch, running_loss / len(train_loader), (time() - time0) / 60
                )
            )


def calculate_accuracy(val_loader, model):
    correct_count, all_count = 0, 0
    for images, labels in val_loader:
        if cuda.is_available():
            labels = labels.cuda()
            images = images.cuda()

        with torch.no_grad():
            output = model(images)

        for i in range(len(labels)):
            logps = output[i]

            ps = torch.exp(logps)
            probab = list(ps.cpu().numpy())
            pred_label = probab.index(max(probab))
            true_label = labels.cpu().numpy()[i]
            if true_label == pred_label:
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("Model Accuracy = {:.2f}%".format(100 * correct_count / all_count))


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def parse_arguments():
    network_arch = "Resnet"

    messages = {
        "ff": "Use Feed Forward network",
        "cnn": "Use Convolutional network",
        "resnet": "Use pretrained Resnet network",
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--ff", help=messages["ff"], action="store_true")
    parser.add_argument("--cnn", help=messages["cnn"], action="store_true")
    parser.add_argument("--resnet", help=messages["resnet"], action="store_true")
    args = parser.parse_args()
    if args.ff:
        network_arch = "FF"
        print(messages["ff"])
    if args.cnn:
        network_arch = "CNN"
        print(messages["cnn"])
    if args.resnet:
        network_arch = "Resnet"
        print(messages["resnet"])

    return network_arch


if __name__ == "__main__":

    arch = parse_arguments()

    train_loader, val_loader = get_data_loaders()

    model = None
    if arch == "FF":
        model = FeedForward()
    if arch == "CNN":
        model = ConvNet()
    if arch == "Resnet":
        # get pretrained model
        model = models.resnet18(pretrained=True)

        # setup first layer with only 1 chanel
        orig_state = model.state_dict()
        orig_weights = copy.deepcopy(orig_state["conv1.weight"].data)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1.weight.data = orig_weights.resize_as_(model.conv1.weight.data)

        # freeze weights
        freeze_model(model)

        # setup last layer with current number of classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    if cuda.is_available():
        model = model.cuda()

    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    train_model(train_loader, model, criterion, epochs=DEFAULT_EPOCHS)
    torch.save(model, "./data/my_mnist_model.pt")

    calculate_accuracy(val_loader, model)

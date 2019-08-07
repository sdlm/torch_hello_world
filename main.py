import torch
import torchvision
from time import time
from torch import nn, optim, cuda
from torch.nn import functional as F


PATH_TO_STORE_TRAINSET = './data/train'
PATH_TO_STORE_TESTSET = './data/test'

IMG_WIDTH = IMG_HEIGHT = 28

INPUT_SIZE = IMG_WIDTH * IMG_HEIGHT
FIRST_HIDDEN_LAYER_SIZE = 128
SECOND_HIDDEN_LAYER_SIZE = 64
OUTPUT_SIZE = 10

DEFAULT_EPOCHS = 15


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
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    # define model
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.LogSoftmax(dim=1))

    return model


class FeedForward(torch.nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_sizes[0])
        self.linear2 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.linear3 = torch.nn.Linear(hidden_sizes[1], output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x0 = x.reshape(x.size(0), -1)  # flat
        x1 = F.relu(self.linear1(x0))  # .clamp(min=0)
        x2 = F.relu(self.linear2(x1))  # .clamp(min=0)
        x3 = self.linear3(x2)
        return self.softmax(x3)


class ConvNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # 26 * 26
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 13 * 13
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # 11 * 11
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 5 * 5
        )
        self.drop_out = nn.Dropout(p=.5)
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


def train_model(train_loader, model, loss_f, epochs):
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    # epochs = 15
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            # Flatten MNIST images into a 784 long vector
            # images = images.view(images.shape[0], -1)
            if cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            # Training pass
            optimizer.zero_grad()

            output = model(images)
            loss = loss_f(output, labels)

            # This is where the model learns by backpropagating
            loss.backward()

            # And optimizes its weights here
            optimizer.step()

            running_loss += loss.item()
        else:
            print(
                "Epoch{:> 3} - Train loss: {:.5f} - Elapsed: {:.2f} min".format(
                    e, running_loss / len(train_loader), (time() - time0) / 60
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
            # img = images[i].view(1, 784)
            # img = images[i]
            # if cuda.is_available():
            #     img = img.cuda()
            # with torch.no_grad():
            #     logps = model(img)
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


if __name__ == '__main__':

    train_loader, val_loader = get_data_loaders()

    model = ConvNet()
    # model = FeedForward(
    #     input_size=INPUT_SIZE,
    #     hidden_sizes=[FIRST_HIDDEN_LAYER_SIZE, SECOND_HIDDEN_LAYER_SIZE],
    #     output_size=OUTPUT_SIZE
    # )
    # model = get_model()
    if cuda.is_available():
        model = model.cuda()

    loss_f = nn.NLLLoss()
    # loss_f = nn.CrossEntropyLoss()

    train_model(train_loader, model, loss_f, epochs=10)  # DEFAULT_EPOCHS
    torch.save(model, './data/my_mnist_model.pt')

    calculate_accuracy(val_loader, model)

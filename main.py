import torch
import torchvision
from time import time
from torch import nn, optim


PATH_TO_STORE_TRAINSET = './data/train'
PATH_TO_STORE_TESTSET = './data/test'


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


def train_model(train_loader, model, loss_f):
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    epochs = 15
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

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
            print("Epoch {} - Training loss: {:.5f}".format(e, running_loss / len(train_loader)))
            print("Training Time {:.2f} min".format((time() - time0) / 60))


def calculate_accuracy(val_loader, model):
    correct_count, all_count = 0, 0
    for images, labels in val_loader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if true_label == pred_label:
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("Model Accuracy =", (correct_count / all_count))


if __name__ == '__main__':

    train_loader, val_loader = get_data_loaders()

    model = get_model()

    loss_f = nn.NLLLoss()

    train_model(train_loader, model, loss_f)
    torch.save(model, './data/my_mnist_model.pt')

    calculate_accuracy(val_loader, model)

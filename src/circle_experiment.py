import copy
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn, optim, cuda
from torch.utils import data
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

from src import classes

DIAMETER = 2
IMG_SIZE = 32
MARGIN = 4
TRAIN_COUNT = 5000
TEST_COUNT = 500
EPOCHS_COUNT = 10


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_image(x: int, y: int) -> Image:
    image = Image.new('L', (IMG_SIZE, IMG_SIZE))
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), image.size], fill='black')
    coords = (x, y, x + DIAMETER, y + DIAMETER)
    draw.ellipse(coords, fill='white')
    return image


def generate_dataset(count: int):
    max_coord = IMG_SIZE - 2 * MARGIN
    x_arr = MARGIN + np.random.randint(max_coord, size=count)
    y_arr = MARGIN + np.random.randint(max_coord, size=count)
    images = [
        generate_image(x_arr[i], y_arr[i])
        for i in range(count)
    ]
    arr = np.array([
        np.array(im)
        for im in images
    ])
    x_center = x_arr + DIAMETER / 2
    return arr, x_center


class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, count: int):
        """Initialization"""
        volatility = IMG_SIZE - 2 * MARGIN
        x_arr = np.random.randint(volatility, size=count) + MARGIN
        y_arr = np.random.randint(volatility, size=count) + MARGIN
        to_tensor = transforms.ToTensor()
        self.values = [to_tensor(generate_image(x_arr[i], y_arr[i])) for i in range(count)]
        x_center = torch.from_numpy(x_arr + DIAMETER / 2).float()
        y_center = torch.from_numpy(y_arr + DIAMETER / 2).float()
        self.labels = [(x_center[i], y_center[i]) for i in range(count)]

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.labels)

    def __getitem__(self, index):
        """Generates one sample of data"""
        return self.values[index], self.labels[index]


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:  # , 'val'
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            max_x_coord = 0
            max_true_x_coord = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # if cuda.is_available():
                #     inputs = inputs.cuda()
                #     labels = labels.cuda()
                inputs = inputs.to(device)
                labels = [l.to(device) for l in labels]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    x_coord = model(inputs)  # , y_coord
                    max_x_coord = max(max_x_coord, x_coord.max().item())
                    x_loss = criterion(x_coord, labels[0].unsqueeze(1))
                    max_true_x_coord = max(max_true_x_coord, labels[0].max().item())
                    # y_loss = criterion(y_coord, labels[1].unsqueeze(1))
                    # joint_loss = 0.5 * (x_loss + y_loss)

                    # _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        x_loss.backward()
                        optimizer.step()
                        scheduler.step(epoch)

                # statistics
                running_loss += x_loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print(
                'Epoch{:> 3}, {} Loss: {:.4f}, max_x_coord = {:.1f}, max_true_x_coord = {:.1f}, fc2.max = {:.2f}'.format(
                    epoch, phase, epoch_loss, max_x_coord, max_true_x_coord, model.fc2.weight.max().item()
                )
            )

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == 'val':
            #     val_acc_history.append(epoch_acc)

        # print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == '__main__':

    dataset_params = {
        'train': TRAIN_COUNT,
        'val': TEST_COUNT,
    }
    image_datasets = {
        x: Dataset(dataset_params[x])
        for x in ['train', 'val']
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }

    model = classes.ConvNet(fc=1).to(device)

    if cuda.is_available():
        print('USE GPU')
        # model = model.cuda()

    criterion = nn.L1Loss()  # L1Loss  MSELoss
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=1000)
    # torch.save(model, './data/my_mnist_model.pt')

    # calculate_accuracy(val_loader, model)
    # train_model(train_loader, model, criterion, optimizer, scheduler, epochs=EPOCHS_COUNT)

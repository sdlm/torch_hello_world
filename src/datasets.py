import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils import data
from torchvision.transforms import transforms
import math


class CircleDataset(data.Dataset):
    """Circle in black background"""

    IMG_SIZE = 32
    DIAMETER = 2
    MARGIN = 4

    def __init__(self, count: int):
        volatility = self.IMG_SIZE - 2 * self.MARGIN
        x_arr = np.random.randint(volatility, size=count) + self.MARGIN
        y_arr = np.random.randint(volatility, size=count) + self.MARGIN
        to_tensor = transforms.ToTensor()
        self.values = [to_tensor(self.generate_image(x_arr[i], y_arr[i])) for i in range(count)]
        x_center = torch.from_numpy(x_arr + self.DIAMETER / 2).float()
        y_center = torch.from_numpy(y_arr + self.DIAMETER / 2).float()
        self.labels = [(x_center[i], y_center[i]) for i in range(count)]

    def generate_image(self, x: int, y: int) -> Image:
        image = Image.new("L", (self.IMG_SIZE, self.IMG_SIZE))
        draw = ImageDraw.Draw(image)
        draw.rectangle([(0, 0), image.size], fill="black")
        coords = (x, y, x + self.DIAMETER, y + self.DIAMETER)
        draw.ellipse(coords, fill="white")
        return image

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        """Generates one sample of data"""
        return self.values[index], self.labels[index]


class RectangleDataset(data.Dataset):
    """Rectangle in black background"""

    IMG_SIZE = 128
    WIDTH = 13
    HEIGHT = 8
    MARGIN = math.sqrt((WIDTH / 2) ** 2 + (HEIGHT / 2) ** 2) + 5
    MAX_ANGLE = 30

    def __init__(self, count: int):
        volatility = self.IMG_SIZE - 2 * self.MARGIN
        x_center = np.random.randint(volatility, size=count) + self.MARGIN
        y_center = np.random.randint(volatility, size=count) + self.MARGIN
        angle = np.random.randint(self.MAX_ANGLE, size=count)
        to_tensor = transforms.ToTensor()
        self.values = [to_tensor(self.generate_image(x_center[i], y_center[i], angle[i])) for i in range(count)]
        self.labels = [None for _ in range(count)]
        for i in range(count):
            coords = self.get_rectangle_coords(x_center[i], y_center[i], angle[i]).reshape((8,))
            self.labels[i] = [float(coords.item(k)) for k in range(8)]
        # self.labels = [
        #     tuple(map(tuple, self.get_rectangle_coords(x_center[i], y_center[i], angle[i]).reshape((8,))))
        #     for i in range(count)
        # ]
        # torch.from_numpy(self.get_rectangle_coords(x_center[i], y_center[i], angle[i])).float()

    def generate_image(self, x: float, y: float, theta: int) -> Image:
        image = Image.new("L", (self.IMG_SIZE, self.IMG_SIZE))
        draw = ImageDraw.Draw(image)
        draw.rectangle([(0, 0), image.size], fill="black")
        draw.rectangle(self.rectangle_at_center, fill="white")
        image = image.rotate(theta, resample=Image.BILINEAR, expand=False)
        center = self.IMG_SIZE / 2
        return self.shift_image(image, x - center, y - center)

    @property
    def rectangle_at_center(self):
        center = self.IMG_SIZE / 2
        return [
            (center - self.WIDTH / 2, center - self.HEIGHT / 2),
            (center + self.WIDTH / 2, center + self.HEIGHT / 2),
        ]

    @staticmethod
    def shift_image(img, shift_x, shift_y):
        return img.transform(img.size, Image.AFFINE, (1, 0, -shift_x, 0, 1, -shift_y))

    def get_rectangle_coords(self, x: int, y: int, theta: int) -> np.matrix:
        a = theta / 180.0 * np.pi
        rotation_matrix = np.matrix(((np.cos(a), -np.sin(a)), (np.sin(a), np.cos(a))))
        coords = self.get_straight_rectangle_coords(0, 0) @ rotation_matrix
        coords[:, 0] = coords[:, 0] + x
        coords[:, 1] = coords[:, 1] + y
        return coords

    def get_straight_rectangle_coords(self, x: float, y: float) -> np.matrix:
        return np.matrix(
            (
                (x + self.WIDTH / 2, y + self.HEIGHT / 2),
                (x + self.WIDTH / 2, y - self.HEIGHT / 2),
                (x - self.WIDTH / 2, y + self.HEIGHT / 2),
                (x - self.WIDTH / 2, y - self.HEIGHT / 2),
            )
        )

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        """Generates one sample of data"""
        return self.values[index], self.labels[index]

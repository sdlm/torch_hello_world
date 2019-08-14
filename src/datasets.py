import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils import data
from torchvision.transforms import transforms

IMG_SIZE = 32


class CircleDataset(data.Dataset):
    """Circle in black background"""

    DIAMETER = 2
    MARGIN = 4

    def __init__(self, count: int):
        volatility = IMG_SIZE - 2 * self.MARGIN
        x_arr = np.random.randint(volatility, size=count) + self.MARGIN
        y_arr = np.random.randint(volatility, size=count) + self.MARGIN
        to_tensor = transforms.ToTensor()
        self.values = [to_tensor(self.generate_image(x_arr[i], y_arr[i])) for i in range(count)]
        x_center = torch.from_numpy(x_arr + self.DIAMETER / 2).float()
        y_center = torch.from_numpy(y_arr + self.DIAMETER / 2).float()
        self.labels = [(x_center[i], y_center[i]) for i in range(count)]

    def generate_image(self, x: int, y: int) -> Image:
        image = Image.new("L", (IMG_SIZE, IMG_SIZE))
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

    WIDTH = 8
    HEIGHT = 5
    MARGIN = 10
    MAX_ANGLE = 22.5

    def __init__(self, count: int):
        volatility = IMG_SIZE - 2 * self.MARGIN
        x_center = np.random.randint(volatility, size=count) + self.MARGIN
        y_center = np.random.randint(volatility, size=count) + self.MARGIN
        angle = np.random.rand((count,)) * self.MAX_ANGLE
        to_tensor = transforms.ToTensor()
        self.values = [to_tensor(self.generate_image(x_center[i], y_center[i], angle[i])) for i in range(count)]
        self.labels = [self.get_rectangle_coords(x_center[i], y_center[i], angle[i]) for i in range(count)]

    def generate_image(self, x: float, y: float, theta: float) -> Image:
        image = Image.new("L", (IMG_SIZE, IMG_SIZE))
        draw = ImageDraw.Draw(image)
        draw.rectangle([(0, 0), image.size], fill="black")

        straight_rectangle_coords = self.get_straight_rectangle_coords(x, y)
        draw.rectangle([(0, 0), image.size], fill="black")

        image = image.rotate(theta, resample=Image.BICUBIC, expand=False)

        coords = (x, y, x + self.DIAMETER, y + self.DIAMETER)
        draw.ellipse(coords, fill="white")
        return image

    def get_rectangle_coords(self, x: float, y: float, theta: float) -> np.ndarray:
        rotation_matrix = np.matrix(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
        return self.get_straight_rectangle_coords(x, y) @ rotation_matrix

    def get_straight_rectangle_coords(self, x: float, y: float) -> np.ndarray:
        return np.array(
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

import os
import sys
import torchvision
from torchvision import transforms

from utils.constants import DATA_PATH
from utils.enums import RetrieveDataType

transform_train_noise = transforms.Compose(
    [
        transforms.CenterCrop(24),
        transforms.Resize(size=32),
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=(3, 7), sigma=(1.1, 2.2)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test_noise = transforms.Compose(
    [
        transforms.CenterCrop(24),
        transforms.Resize(size=32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_train_noise_rotated = transforms.Compose(
    [
        torchvision.transforms.RandomRotation((30, 30)),
        transforms.CenterCrop(24),
        transforms.Resize(size=32),
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=(3, 7), sigma=(1.1, 2.2)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test_noise_rotated = transforms.Compose(
    [
        torchvision.transforms.RandomRotation((30, 30)),
        transforms.CenterCrop(24),
        transforms.Resize(size=32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


class DataRetriever:
    def __init__(self, base_directory: str = DATA_PATH):
        self._base_directory = base_directory
        os.makedirs(self._base_directory, exist_ok=True)

        # Initialize dataset placeholders
        self.train_original = None
        self.test_original = None
        self.train_noise = None
        self.test_noise = None
        self.train_noise_rotated = None
        self.test_noise_rotated = None

    def get_data(self, retrieve_data_type: RetrieveDataType):
        match retrieve_data_type:
            case RetrieveDataType.TRAIN_ORIGINAL:
                if self.train_original is None:
                    self.train_original = torchvision.datasets.CIFAR10(
                        root=self._base_directory, train=True, download=True
                    )
                return self.train_original

            case RetrieveDataType.TEST_ORIGINAL:
                if self.test_original is None:
                    self.test_original = torchvision.datasets.CIFAR10(
                        root=self._base_directory, train=False, download=True
                    )
                return self.test_original

            case RetrieveDataType.TRAIN_NOISE:
                if self.train_noise is None:
                    self.train_noise = torchvision.datasets.CIFAR10(
                        root=self._base_directory,
                        train=True,
                        download=True,
                        transform=transform_train_noise,
                    )
                return self.train_noise

            case RetrieveDataType.TEST_NOISE:
                if self.test_noise is None:
                    self.test_noise = torchvision.datasets.CIFAR10(
                        root=self._base_directory,
                        train=False,
                        download=True,
                        transform=transform_test_noise,
                    )
                return self.test_noise

            case RetrieveDataType.TRAIN_NOISE_ROTATED:
                if self.train_noise_rotated is None:
                    self.train_noise_rotated = torchvision.datasets.CIFAR10(
                        root=self._base_directory,
                        train=True,
                        download=True,
                        transform=transform_train_noise_rotated,
                    )
                return self.train_noise_rotated

            case RetrieveDataType.TEST_NOISE_ROTATED:
                if self.test_noise_rotated is None:
                    self.test_noise_rotated = torchvision.datasets.CIFAR10(
                        root=self._base_directory,
                        train=False,
                        download=True,
                        transform=transform_test_noise_rotated,
                    )
                return self.test_noise_rotated

            case _:
                raise ValueError("Invalid data retrieval type")

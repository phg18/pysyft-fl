#1. Data-owner-upload-data

# third party
import matplotlib.pyplot as plt

# relative import
from mnist_dataset import mnist
from mnist_dataset import mnist_raw
import numpy as np

# syft absolute
import syft as sy
import torch 

print(f"{sy.__version__ = }")
print(f"{torch.__version__ = }")


server = sy.orchestra.launch(name="mnist-torch-datasite", dev_mode=True, reset=True)
root_client = server.login(email="info@openmined.org", password="changethis")
train_images, train_labels, _, _ = mnist_raw()

print(f"{train_images.shape = }")
print(f"{train_labels.shape = }")
train_images, train_labels, _, _ = mnist()
num_samples = 1000
train_images = train_images[:num_samples, :]
train_labels = train_labels[:num_samples, :]

print(f"{train_images.shape = }")
print(f"{train_labels.shape = }")

mock_images = np.random.rand(num_samples, 784)
mock_labels = np.eye(10)[np.random.choice(10, num_samples)]

assert mock_labels.shape == train_labels.shape
assert mock_images.shape == train_images.shape
dataset = sy.Dataset(
    name="MNIST data",
    description="""Contains the flattened training images and one-hot encoded training labels.""",
    url="https://storage.googleapis.com/cvdf-datasets/mnist/",
)

dataset.add_contributor(
    role=sy.roles.UPLOADER,
    name="Alice",
    email="alice@openmined.com",
    note="Alice is the data engineer at the OpenMined",
)

asset_mnist_train_input = sy.Asset(
    name="MNIST training images",
    description="""The training images of the MNIST dataset""",
    data=train_images,
    mock=mock_images,
)

asset_mnist_train_labels = sy.Asset(
    name="MNIST training labels",
    description="""The training labels of MNIST dataset""",
    data=train_labels,
    mock=mock_labels,
)

dataset.add_asset(asset_mnist_train_input)
dataset.add_asset(asset_mnist_train_labels)
root_client.upload_dataset(dataset)
datasets = root_client.api.services.dataset.get_all()

assert len(datasets) == 1
datasets
datasets[0].assets[0]
datasets[0].assets[1]
register_result = root_client.register(
    name="Pablo Herrero",
    email="phg18@alu.ua.es",
    password="123456Ab",
    password_verify="123456Ab",
    institution="Universidad de Alicante",
    website="https://www.ua.es/",
)
assert isinstance(register_result, sy.SyftSuccess)
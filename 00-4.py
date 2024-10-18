#4. Data-scientist-download-results

# third party
from mnist_dataset import mnist
import torch

# syft absolute
import syft as sy

server = sy.orchestra.launch(name="mnist-torch-datasite", dev_mode=True)
ds_client = server.login(email="phg18@alu.ua.es", password="123456Ab")

datasets = ds_client.datasets.get_all()
assets = datasets[0].assets
assert len(assets) == 2

training_images = assets[0]
training_labels = assets[1]

ds_client.code

result = ds_client.code.mnist_3_linear_layers_torch(
    mnist_images=training_images, mnist_labels=training_labels
)

train_accs, params = result.get_from(ds_client)

assert isinstance(train_accs, list)
train_accs

assert isinstance(params, dict)
params

_, _, test_images, test_labels = mnist()

assert test_images.shape == (10000, 784)
assert test_labels.shape == (10000, 10)

# third party
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.log_softmax(self.fc3(x), dim=1)
        return x


# Print the model to see the architecture
model = MLP()

def accuracy(model, batch, params=None):
    if params is not None:
        model.load_state_dict(params)

    # Convert inputs and targets to PyTorch tensor
    inputs, targets = batch
    inputs = torch.tensor(inputs)
    targets = torch.tensor(targets)

    # Get model predictions
    with torch.no_grad():
        outputs = model(inputs)
        print(outputs.shape)
    # Get predicted class
    _, predicted_class = torch.max(outputs, dim=1)
    print(predicted_class.shape)

    # Calculate accuracy
    accuracy = torch.mean((predicted_class == torch.argmax(targets, dim=1)).float())
    return accuracy.item()  # Convert accuracy to a Python scalar

test_acc = accuracy(model, (test_images, test_labels))
print(f"Test set accuracy with random weights = {test_acc * 100 : .2f}%")

test_acc = accuracy(model, (test_images, test_labels), params)
print(f"Test set accuracy with trained weights = {test_acc * 100 : .2f}%")

assert test_acc * 100 > 70
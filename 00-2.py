#2. Data-scientist-submit-code

# third party
import matplotlib.pyplot as plt
import numpy as np

# syft absolute
import syft as sy

server = sy.orchestra.launch(name="mnist-torch-datasite", dev_mode=True)
ds_client = server.login(email="phg18@alu.ua.es", password="123456Ab")

datasets = ds_client.datasets.get_all()
assert len(datasets) == 1
datasets
assets = datasets[0].assets
assert len(assets) == 2
assets
training_images = assets[0]
training_images
training_labels = assets[1]
training_labels
assert training_images.data is None
training_labels.data
mock_images = training_images.mock
plt.imshow(np.reshape(mock_images[0], (28, 28)))
mock_images_ptr = training_images.pointer
mock_images_ptr
type(mock_images_ptr)
mock_labels = training_labels.mock
mock_labels_ptr = training_labels.pointer
mock_labels_ptr

@sy.syft_function(
    input_policy=sy.ExactMatch(
        mnist_images=mock_images_ptr, mnist_labels=mock_labels_ptr
    ),
    output_policy=sy.SingleExecutionExactOutput(),
)
def mnist_3_linear_layers_torch(mnist_images, mnist_labels):
    # third party
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset

    # Convert NumPy arrays to PyTorch tensors
    images_tensor = torch.tensor(mnist_images, dtype=torch.float32)
    labels_tensor = torch.tensor(mnist_labels, dtype=torch.float32)
    # Create a PyTorch dataset using TensorDataset
    custom_dataset = TensorDataset(images_tensor, labels_tensor)
    # Define the data loader
    train_loader = torch.utils.data.DataLoader(
        custom_dataset, batch_size=4, shuffle=True
    )

    # Define the neural network class
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

    # Define the model, optimizer, and loss function
    model = MLP()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Function to calculate accuracy
    def accuracy(model, data_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
        return correct / total

    # Train the model
    num_epochs = 20
    train_accs = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for _, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            f"Epoch {epoch + 1}, Loss: {(running_loss / len(train_loader)):.4f}",
            end=". ",
        )
        # Calculate accuracy on the training set
        train_accuracy = accuracy(model, train_loader)
        train_accs.append(train_accuracy)
        print(f"Training set accuracy: {train_accuracy}%")

    # Get model parameters
    params = model.state_dict()

    # Return training accuracy and model parameters
    return train_accs, params

new_project = sy.Project(
    name="Training a 3-layer torch neural network on MNIST data",
    description="""Hi, I would like to train my neural network on your MNIST data 
                (I can download it online too but I just want to use Syft coz it's cool)""",
    members=[ds_client],
)
new_project

new_project.create_code_request(obj=mnist_3_linear_layers_torch, client=ds_client)
ds_client.code
project = new_project.send()
assert isinstance(project, sy.service.project.project.Project)
project.events
project.requests
project.requests[0]
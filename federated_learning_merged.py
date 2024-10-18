# Import necessary libraries
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset
import syft as sy
from mnist_dataset import mnist
import MLP as mlp
import logging
logging.getLogger("syft").setLevel(logging.WARNING)



# Step 1: Load the data and configure the server
server = sy.orchestra.launch(name="mnist-torch-datasite", dev_mode=True, reset=True)

# Data Owner logs into the server
root_client = server.login(email="info@openmined.org", password="changethis")

# Load the MNIST data
train_images, train_labels, _, _ = mnist()
num_samples = 1000
train_images = train_images[:num_samples, :]
train_labels = train_labels[:num_samples, :]

# Simulate mock data
mock_images = np.random.rand(num_samples, 784)
mock_labels = np.eye(10)[np.random.choice(10, num_samples)]

# Create and upload the federated dataset
dataset = sy.Dataset(name="MNIST data", description="Flattened MNIST images and labels")
asset_mnist_train_input = sy.Asset(name="MNIST training images", data=train_images, mock=mock_images)
asset_mnist_train_labels = sy.Asset(name="MNIST training labels", data=train_labels, mock=mock_labels)
dataset.add_asset(asset_mnist_train_input)
dataset.add_asset(asset_mnist_train_labels)
root_client.upload_dataset(dataset)

# Register a new user (scientist)
register_result = root_client.register(
    name="Pablo Herrero",
    email="phg18@alu.ua.es",
    password="123456Ab",
    password_verify="123456Ab",
    institution="Universidad de Alicante",
    website="https://www.ua.es/",
)

# Step 2: Data scientist logs into the server and requests dataset usage
ds_client = server.login(email="phg18@alu.ua.es", password="123456Ab")
datasets = ds_client.datasets.get_all()
training_images = datasets[0].assets[0]
training_labels = datasets[0].assets[1]

# Obtain mock data pointers
mock_images_ptr = training_images.pointer
mock_labels_ptr = training_labels.pointer


# Define the training function
@sy.syft_function(input_policy=sy.ExactMatch(mnist_images=mock_images_ptr, mnist_labels=mock_labels_ptr),
                  output_policy=sy.SingleExecutionExactOutput())
def mnist_3_linear_layers_torch(mnist_images, mnist_labels):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from MLP import MLP
    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset

    
    # Convert data to tensors
    images_tensor = torch.tensor(mnist_images, dtype=torch.float32)
    labels_tensor = torch.tensor(mnist_labels, dtype=torch.float32)
    
    # Create a PyTorch dataset and data loader
    dataset = TensorDataset(images_tensor, labels_tensor)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize model, optimizer, and loss function
    model = MLP()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Function to calculate accuracy
    def accuracy(model, data_loader):
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
                total += labels.size(0)
        return correct / total

    # Train the model
    num_epochs = 20
    train_accs = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_accuracy = accuracy(model, train_loader)
        train_accs.append(train_accuracy)
        print(f"Epoch {epoch+1}, Accuracy: {train_accuracy:.2%}")
    
    # Return model parameters
    return train_accs, model.state_dict()


# Create and send the code request for the project
new_project = sy.Project(name="MNIST Neural Network Training", members=[ds_client])
new_project.create_code_request(obj=mnist_3_linear_layers_torch, client=ds_client)
project = new_project.send()


# Step 3: Data owner approves the request
request = root_client.projects[0].requests[0]
request.approve()


# Step 4: Data scientist retrieves and evaluates the model
_, _, test_images, test_labels = mnist()
result = ds_client.code.mnist_3_linear_layers_torch(
    mnist_images=training_images, mnist_labels=training_labels
)

train_accs, params = result.get_from(ds_client)


# Function to evaluate the model
def evaluate_model(model, data, labels, params=None):
    if params is not None:
        model.load_state_dict(params)
    
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(data_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == torch.argmax(labels_tensor, dim=1)).float().mean().item()
    
    return accuracy


# Test the trained model on test data
model = mlp.MLP()
test_accuracy = evaluate_model(model, test_images, test_labels, params)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

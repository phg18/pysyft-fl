#3. Data-ower-review-approve-code

# syft absolute
import syft as sy
from syft.service.request.request import RequestStatus

print(f"{sy.__version__ = }")

server = sy.orchestra.launch(name="mnist-torch-datasite", dev_mode=True)
root_client = server.login(email="info@openmined.org", password="changethis")

root_client.projects

requests = root_client.projects[0].requests

assert len(requests) == 1

request = requests[0]
assert request.status == RequestStatus.PENDING
request

change = request.changes[0]
change

# gettting a reference to the user code object
user_code = change.code

# viewing the actual code submitted for request
user_code.show_code

assert len(user_code.assets) == 2
user_code.assets

mock_images = user_code.assets[0].mock
print(f"{mock_images.shape = }")
mock_labels = user_code.assets[1].mock
print(f"{mock_labels.shape = }")

users_function = user_code.run
users_function

mock_train_accs, mock_params = users_function(
    mnist_images=mock_images, mnist_labels=mock_labels
)

assert isinstance(mock_train_accs, list)
mock_train_accs

# private data associated with the asset
private_images = user_code.assets[0].data
print(f"{private_images.shape = }")
private_labels = user_code.assets[1].data
print(f"{private_labels.shape = }")

train_accs, params = users_function(
    mnist_images=private_images, mnist_labels=private_labels
)

assert isinstance(train_accs, list)
train_accs

assert isinstance(params, dict)

res = request.approve()

assert isinstance(res, sy.SyftSuccess)
res
params
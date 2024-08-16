import torch
import time
import torchinfo
import torchvision

# Set the device
device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)
print(f"Using {device_type.upper()}...")

# Build the model
BATCH_SIZE = 128

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # First Convolutional Layer with He-Weight Initialization and Batch Normalization
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')  # He-Weight Initialization
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.act1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Second Convolutional Layer with Batch Normalization
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')  # He-Weight Initialization
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.act2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully Connected Layer with Softmax Activation
        self.fc1 = torch.nn.Linear(in_features=64*7*7, out_features=10, bias=True)
        self.fcact1 = torch.nn.Softmax(dim=1)

    def forward(self, input_x):
        assert torch.tensor(input_x.shape[1:]).prod().item() == 784, "input_x must have 784 features"
        assert input_x.shape.__len__() == 4, "input_x must be of rank 4"

        # Forward pass through the convolutional layers
        output_y = self.conv1(input_x)
        output_y = self.bn1(output_y)
        output_y = self.act1(output_y)
        output_y = self.pool1(output_y)
        
        output_y = self.conv2(output_y)
        output_y = self.bn2(output_y)
        output_y = self.act2(output_y)
        output_y = self.pool2(output_y)

        # Flatten the output and pass through fully connected layers
        output_y = output_y.flatten(start_dim=1)
        output_y = self.fc1(output_y)
        output_y = self.fcact1(output_y)

        return output_y

model = CNN().to(device=device)
print(torchinfo.summary(model, (BATCH_SIZE, 1, 28, 28)))

# Create the data pipeline
def transform_x(x):
    x = torchvision.transforms.functional.pil_to_tensor(x)
    x = x / 255.0
    return x

def transform_y(y):
    y = torch.nn.functional.one_hot(torch.tensor(y), num_classes=10)
    return y

training_data = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.Lambda(transform_x),
    target_transform=torchvision.transforms.Lambda(transform_y)
)

test_data = torchvision.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.Lambda(transform_x),
    target_transform=torchvision.transforms.Lambda(transform_y)
)

train_dl = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

# Compile the model using SGD and Cross Entropy Loss
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()

def calc_accuracy(true_y, pred_y, is_count=False):
    pred_y = pred_y.argmax(dim=1)
    true_y = true_y.argmax(dim=1)
    if is_count:
        return (true_y == pred_y).to(torch.float64).sum().item()
    else:
        return (true_y == pred_y).to(torch.float64).mean().item()

# Training loop with 5-Fold Cross Validation
t1 = time.time()
epoch = 10
for ep in range(epoch):
    model.train()  # Set the model to training mode
    for i, (input_x, true_y) in enumerate(train_dl):
        input_x, true_y = input_x.to(device=device), true_y.to(device=device)
        
        # Forward pass
        pred_y = model(input_x)
        
        # Compute loss
        loss = loss_fn(pred_y, true_y.argmax(dim=1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Calculate accuracy
        acc = calc_accuracy(true_y, pred_y)
    print(f"Epoch: {ep+1}, Last Batch Loss: {loss:.2f}, Last Batch Acc: {acc:.2f}")

t2 = time.time()
print(f"Time taken on {device_type}: {(t2-t1):.5f} sec")

# Test accuracy
def calculate_accuracy(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device=device), labels.to(device=device)
            outputs = model(inputs)
            correct += calc_accuracy(true_y=labels, pred_y=outputs, is_count=True)
            total += len(labels)

    # Calculate accuracy
    accuracy = correct / total
    return accuracy

test_accuracy = calculate_accuracy(model, test_dl)
print("Test Accuracy:", test_accuracy)

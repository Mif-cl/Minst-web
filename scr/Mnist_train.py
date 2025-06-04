import torch
from torch import nn,save,load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ImageClassifier import ImageClassifier

# Loading Data
transform = transforms.Compose([transforms.ToTensor()])

# Training data
train_data = datasets.MNIST(root="data", download=True, train=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Test data
test_data = datasets.MNIST(root="data", download=True, train=False, transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageClassifier().to(device)

# Define the optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train the model
for epoch in range(10):  # Train for 10 epochs
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Reset gradients
        outputs = model(images)  # Forward pass
        loss = loss_fn(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

    print(f"Epoch:{epoch} loss is {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), 'checkpoint.pt')
print("Model saved to checkpoint.pt")

# Test loop
def test_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    test_loss = 0
    
    with torch.no_grad():  # Disable gradient computation for testing
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"Test Results:")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return accuracy, avg_test_loss

# Run test
print("\nTesting the model...")
test_accuracy, test_loss = test_model(model, test_loader, device)
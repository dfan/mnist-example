import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 128
learning_rate = 0.0008

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
  def __init__(self, num_classes=10):
    super(ConvNet, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(1, 6, kernel_size=5),
      nn.BatchNorm2d(6),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer2 = nn.Sequential(
      nn.Conv2d(6, 16, kernel_size=5),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer3 = nn.Sequential(
      nn.Conv2d(16, 120, kernel_size=4),
      nn.ReLU())
    self.fc = nn.Sequential(
      nn.Linear(120, 84),
      nn.ReLU(),
      nn.Linear(84, num_classes))
        
  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    return out

model = ConvNet(num_classes).to(device)

def testModel(model):
  # Test the model
  model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
    model.train() # reset model
    return correct / total  

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
iterations = []
curr_iter = 0
losses = []
accuracy = []
for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)
        
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
        
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 50 == 0:
      curr_iter += 50
      iterations.append(curr_iter)
      losses.append(loss.item())
      curr_acc = testModel(model)
      accuracy.append(curr_acc)
      print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item(), curr_acc))


final_acc = testModel(model)
print('Final accuracy: {}'.format(accuracy[len(accuracy) - 1]))
# Plot loss and accuracy curves
plt.plot(iterations, losses, label='Loss')
plt.plot(iterations, accuracy, label='Accuracy')
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss/Accuracy")
plt.savefig('mnist_plot.png')
plt.close()

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

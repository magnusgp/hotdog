import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Hotdog_NotHotdog(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path='/dtu/datasets1/02516/hotdog_nothotdog'):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y
    
class Dataset_Loader():
    def __init__(self, train_batch_size, test_batch_size, image_size=(224,224)):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])])
        
    def get_train_loader(self):
        trainset = Hotdog_NotHotdog(train=True, transform=self.transform)
        train_loader = DataLoader(trainset, batch_size=self.train_batch_size, shuffle=True)
        return train_loader
    
    def get_test_loader(self):
        testset = Hotdog_NotHotdog(train=False, transform=self.transform)
        test_loader = DataLoader(testset, batch_size=self.test_batch_size, shuffle=False)
        return test_loader

class Hotdog_Network(nn.Module):
    def __init__(self):
        super(Hotdog_Network, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
 
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
 
        self.flat = nn.Flatten()
 
        self.fc3 = nn.Linear(8192, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
 
        self.fc4 = nn.Linear(512, 2)
 
    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        x = self.pool2(x)
        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 2
        x = self.fc4(x)
        return x
    
if __name__ == "__main__":
    # Initialize Hydra
    #initialize(config_path="config")

    if torch.cuda.is_available():
        print("Running on GPU...\n")
    else:
        print("Running on CPU...\n")

    # Initialize WandB
    wandb.init(project="hotdog_nothotdog", entity="magnusgp")

    # Load configurations
    #cfg = compose(config_name="config")

    # Load the dataset
    train_batch_size = 64
    test_batch_size = 64
    dataset_loader = Dataset_Loader(train_batch_size, test_batch_size, image_size=(128, 128))
    train_loader = dataset_loader.get_train_loader()
    test_loader = dataset_loader.get_test_loader()

    images, labels = next(iter(train_loader))

    # Instantiate the model
    model = Hotdog_Network().to(device)

    # Define loss function and optimizer using Hydra config
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop using Hydra config
    epochs = 100

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        # Iterate through the dataset
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Log training loss to WandB
            wandb.log({"Training Loss": loss.item()})
            
            # # Print statistics
            # running_loss += loss.item()
            # if i % 100 == 99:  # Print every 100 mini-batches
            #     print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}")
            #     running_loss = 0.0
        if epoch % 10 == 9:
            # Log training accuracy to WandB
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / train_batch_size
            wandb.log({"Training Accuracy": accuracy})

    # Evaluating on the test set based on Hydra config
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the test set: {accuracy:.2f}%")

    # Log final accuracy to WandB
    wandb.log({"Test Accuracy": accuracy})
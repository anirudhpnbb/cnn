# train.py

import torch
import torch.optim as optim
import torch.nn as nn
from data_preprocessing import get_dataloaders
from detection import PneumoniaDetectionCNN

data_dir = "/home/anirudh/Desktop/Projects/cnn/chest_xray"
train_loader, val_loader, _ = get_dataloaders(data_dir)

model = PneumoniaDetectionCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, criterion, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

train(model, train_loader, criterion, optimizer, num_epochs=25)
torch.save(model.state_dict(), "pneumonia_detection_model.pth")

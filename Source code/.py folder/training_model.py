import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report

def train_model(mlp_model, train_loader, criterion, optimizer, device, num_epochs=5):
    for epoch in range(num_epochs):
        mlp_model.train()
        train_loss = 0.0

        for batch in train_loader:
            embeddings = batch['embedding'].to(device)
            labels = batch['labels'].to(device)
            embeddings = torch.mean(embeddings, dim=1)
            
            optimizer.zero_grad()
            outputs = mlp_model(embeddings)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss}")

def evaluate_model(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            embeddings = batch['embedding'].to(device)
            labels = batch['labels'].to(device)
            embeddings = torch.mean(embeddings, dim=1)

            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

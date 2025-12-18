#!/usr/bin/env python3
"""
Model training example using torchada.

This example demonstrates training a simple neural network
that works on both CUDA and MUSA platforms transparently.

Usage:
    Just import torchada at the top of your script, then use
    torch.cuda.* APIs as you normally would. torchada patches
    PyTorch to transparently redirect to MUSA on Moore Threads hardware.
"""

import torchada  # noqa: F401 - Import first to apply patches (must be before torch.cuda usage)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Use standard torch.cuda imports - torchada patches them to work on MUSA
from torch.cuda.amp import autocast, GradScaler


class SimpleModel(nn.Module):
    """A simple MLP model."""

    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_epoch(model, dataloader, criterion, optimizer, scaler, use_amp=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        # Move to GPU (works on both CUDA and MUSA)
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        if use_amp:
            # Mixed precision training
            with autocast():
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    # Check GPU availability - use standard torch.cuda API
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name()}")

    # Create dummy dataset
    num_samples = 10000
    input_size = 784
    num_classes = 10

    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Create model, move to GPU
    model = SimpleModel(input_size=input_size, output_size=num_classes)
    model = model.cuda()  # Works on both CUDA and MUSA

    print(f"Model device: {next(model.parameters()).device}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    # Train for a few epochs
    num_epochs = 3
    use_amp = torch.cuda.is_available()  # Use AMP only on GPU

    print(f"\nTraining for {num_epochs} epochs (AMP: {use_amp})...")

    for epoch in range(num_epochs):
        loss = train_epoch(model, dataloader, criterion, optimizer, scaler, use_amp)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

        if torch.cuda.is_available():
            print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    print("\nTraining complete!")

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


"""
This file realise the Convolutional Neural Network deep learning classification model with the help of PyTorch.
CNN is used in binary and multiple classification tasks of MRI images.
"""
# PyTorch packages
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import numpy as np
from sklearn.model_selection import train_test_split

# Progress bar
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# ----------------------------------------------------------------------------------------------------------------------
# Construct dataset.


# Fit our dataset as the form of numpy matrix with torch's DataLoader.
# To inherit torch's Dataset method in a new class, we must overwrite __getitem__()和__len__() methods.
import PreProcessing


class GetTorchData(torch.utils.data.Dataset):
    def __init__(self, data, label, transform):
        """
            Feed our numpy-format dataset when initialization

            Inputs
                data: numpy matrix format dataset
                label: numpy vector label.
                transform: Used to preprocess the inpiut image data.
        """
        self.data = data
        self.label = torch.LongTensor(label)
        self.transform = transform

    def __getitem__(self, index):
        """
        Inputs
            index: indices acquired after dividing the dataset according to batch_size.
        """
        data = self.data[index]
        data = self.transform(data)

        labels = self.label[index]
        return data, labels

    def __len__(self):
        """
        Return
            Length of dataset. This return value is required by DataLoader.
        """
        return len(self.data)


def GetTorchDataLoader(dataset, batch_size):
    """
    Get the PyTorch data loader of trainset or valid set.
    
    Return:
        PyTorch data loader.
    """
    # You may adjust the parameter num_workers according to the CPU cores number of your computer.
    # Bugs appear when num_workers>0 using multi cores on my computer. So I use num_workers = 0.
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return data_loader


# ----------------------------------------------------------------------------------------------------------------------
# Construct CNN.
# The feature maps must be flattened into a vector as input to last fully connected layer.

class CNN(nn.Module):
    def __init__(self, is_mul):
        super().__init__()

        # Each MRI image is (512,512) size.
        self.cnn_layer = nn.Sequential(
            # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            # MaxPool2D(kernel_size, stride, padding)
            nn.Conv2d(1, 64, 3, 1, 1),  # Now it's 64 * 512 * 512
            nn.BatchNorm2d(64),  # Normalize
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),  #

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0)

        )

        if is_mul:
            self.fully_conn = nn.Sequential(
                nn.Linear(128 * 32 * 32, 4),
                nn.BatchNorm1d(4)

                # nn.ReLU(),
                # nn.Linear(1024, 256),
                # nn.ReLU(),
                # nn.Linear(256, 4)
            )
        else:
            self.fully_conn = nn.Sequential(
                nn.Linear(512 * 8 * 8, 2),
                nn.BatchNorm1d(2)
                # nn.ReLU(),
                # nn.Linear(1024, 256),
                # nn.ReLU(),
                # nn.Linear(256, 2)
            )

    def forward(self, x):
        x = self.cnn_layer(x)
        x = x.flatten(1)
        x = self.fully_conn(x)
        return x
        # return F.log_softmax(x, dim=1)


# ----------------------------------------------------------------------------------------------------------------------
# Training and Validation process.
def train_valid_model(train_loader, valid_loader, epoch_num, is_mul):
    # ------------------------------------------------------------------------------------------------------------------
    # Training process.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create Tensorboard writer
    writer = SummaryWriter()
    print("Tensorboard summary writer created.")

    # Initialize model and put it on cpu.
    model = CNN(is_mul=is_mul)
    model.to(device)
    print(model)

    # Use cross entropy as the loss function.
    criterion = nn.CrossEntropyLoss()

    # Use Adam as optimizer. Manually set learning rate and fine tune it with experiments.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Set the epoch numbers
    epoch_num = epoch_num

    # Set a seed to store the states files of model.
    seed = torch.initial_seed()
    print('Use seed : {}'.format(seed))

    for epoch in range(epoch_num):
        # Model turns to train mode.
        model.train()

        train_loss = []
        train_accu = []

        for batch in tqdm(train_loader):
            # Each batch in torch consists of data and labels.
            data, labels = batch

            # Add one "batch" dimension to data's 1th dimension
            # data = torch.unsqueeze(data, 1)

            # Output the calculated result.
            res = model(data.float().to(device))

            # Calculate loss
            loss = criterion(res, labels.to(device))

            # Optimization
            # Gradients params in previous step should be cleared out.
            optimizer.zero_grad()
            # Backward
            loss.backward()

            # Normalization of gradients to avoid too big number.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update gradients.
            optimizer.step()

            # Accuracy
            # dim = -1 means find the index of max number by columns
            accu = (res.argmax(dim=-1) == labels.to(device)).float().mean()

            train_loss.append(loss.item())
            train_accu.append(accu)

        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_train_accu = sum(train_accu) / len(train_accu)

        writer.add_scalar('train loss', avg_train_loss, epoch + 1)
        writer.add_scalar('train accuracy', avg_train_accu, epoch + 1)

        print(f"[ Train | {(epoch + 1) : 05d} ] loss = {avg_train_loss:.6f}, accu = {avg_train_accu:.6f}")

        # ------------------------------------------------------------------------------------------------------------------
        # Validation process.

        # Model turns to evaluate mode.
        model.eval()

        valid_loss = []
        valid_accu = []

        for batch in tqdm(valid_loader):
            data, labels = batch
            # data = torch.unsqueeze(data, 1)

            # No back propagation in valid process.
            # Use torch.no_grad() to avoid using gradient.
            with torch.no_grad():
                res = model(data.float().to(device))

            loss = criterion(res, labels.to(device))

            accu = (res.argmax(dim=-1) == labels.to(device)).float().mean()

            valid_loss.append(loss.item())
            valid_accu.append(accu)

        avg_valid_loss = sum(valid_loss) / len(valid_loss)
        avg_valid_accu = sum(valid_accu) / len(valid_accu)

        writer.add_scalar('valid loss', avg_valid_loss, epoch + 1)
        writer.add_scalar('valid accuracy', avg_valid_accu, epoch + 1)

        print(f"[ Valid | {(epoch + 1) : 05d}] loss = {avg_valid_loss:.6g}, accu = {avg_valid_accu:.5f}")

        # Save the model states

        state = {
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'seed': seed
        }
        model_states_path = os.path.join("model_states_tmp", "model_states_epoch{}".format(epoch + 1))
        torch.save(state, model_states_path)


if __name__ == "__main__":
    x_train, x_valid, y_train, y_valid = PreProcessing.gen_train_test_set(is_mul=True, random_state=108)
    # Convert into torch data loaders.

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.ToTensor()
    ])

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor
    ])

    torch_train_data = GetTorchData(x_train.reshape(2400, 512, 512), y_train, train_transform)
    torch_valid_data = GetTorchData(x_valid.reshape(600, 512, 512), y_valid, valid_transform)

    # Set batch size which will be used in training, validation and testing.
    batch_size = 2

    torch_train_loader = GetTorchDataLoader(torch_train_data, batch_size)
    torch_valid_loader = GetTorchDataLoader(torch_valid_data, batch_size)
    epoch_num = 10

    # For test --------------------------------------------------------------------------------------------------------
    # import matplotlib.pyplot as plt
    # for i, data in enumerate(torch_valid_loader):
    #     print(f"Batch {i} \n")
    #     plt.imshow(data[0][0], cmap='gray')
    #     plt.show()
    # -----------------------------------------------------------------------------------------------------------------

    train_valid_model(torch_train_loader, torch_valid_loader, epoch_num, is_mul=True)

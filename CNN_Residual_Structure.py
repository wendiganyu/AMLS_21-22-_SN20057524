"""
This file realise the Convolutional Neural Network deep learning classification model with the help of PyTorch.
CNN is used in binary and multiple classification tasks of MRI images.
"""
import argparse
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm.auto import tqdm  # Progress bar in training progress
import PreProcessing

# Optional packages for saving images after data augmentation in the training process.
'''
from torch.utils.tensorboard import SummaryWriter
'''

# ----------------------------------------------------------------------------------------------------------------------
# Tran dataset.

# Fit our dataset as the form of numpy matrix with torch's DataLoader.
# To inherit torch's Dataset method in a new class, we must overwrite __getitem__()和__len__() methods.
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
        data = self.transform(np.uint8(data))

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
class Residual_unit(nn.Module):
    """
    Define a unit which can be repeated used of Residual CNN network.
    """

    def __init__(self, input_channel_1, input_channel_2, output_channel, stride_1=1):
        super(Residual_unit, self).__init__()

        # input_channel_2 == output_channel_2 == input_channel_3

        self.conv1 = nn.Conv2d(input_channel_1, input_channel_2, kernel_size=1, stride=stride_1)
        self.batch_normal1 = nn.BatchNorm2d(input_channel_2)

        self.conv2 = nn.Conv2d(input_channel_2, input_channel_2, kernel_size=3, padding=1, stride=1)
        self.batch_normal2 = nn.BatchNorm2d(input_channel_2)

        self.conv3 = nn.Conv2d(input_channel_2, output_channel, kernel_size=1, stride=1)
        self.batch_normal3 = nn.BatchNorm2d(output_channel)

        self.x_1x1_conv = nn.Conv2d(input_channel_1, output_channel, kernel_size=1, stride=stride_1)

    def forward(self, X):
        Y = F.relu(self.batch_normal1(self.conv1(X)))
        Y = F.relu(self.batch_normal2(self.conv2(Y)))
        Y = self.batch_normal3(self.conv3(Y))

        X = self.x_1x1_conv(X)

        Y += X
        return F.relu(Y)


class CNN(nn.Module):
    def __init__(self, is_mul):
        super().__init__()

        b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=13, stride=4, padding=6),  # 128 * 128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64 * 64
        )

        blk2 = []
        # Residual_unit(input_channel_1, input_channel_2, output_channel, stride_1=1):
        blk2.append(Residual_unit(64, 64, 256))
        blk2.append(Residual_unit(256, 64, 256))
        blk2.append(Residual_unit(256, 64, 256))
        b2 = nn.Sequential(*blk2)

        blk3 = []
        blk3.append(Residual_unit(256, 128, 512, stride_1=2))  # 32 * 32
        blk3.append(Residual_unit(512, 128, 512))
        blk3.append(Residual_unit(512, 128, 512))
        blk3.append(Residual_unit(512, 128, 512))
        b3 = nn.Sequential(*blk3)

        blk4 = []
        blk4.append(Residual_unit(512, 256, 1024, stride_1=2))  # 16 * 16
        blk4.append(Residual_unit(1024, 256, 1024))
        blk4.append(Residual_unit(1024, 256, 1024))
        blk4.append(Residual_unit(1024, 256, 1024))
        blk4.append(Residual_unit(1024, 256, 1024))
        blk4.append(Residual_unit(1024, 256, 1024))
        b4 = nn.Sequential(*blk4)

        blk5 = []
        blk5.append(Residual_unit(1024, 512, 2048, stride_1=2))  # 8 * 8
        blk5.append(Residual_unit(2048, 512, 2048))
        blk5.append(Residual_unit(2048, 512, 2048))
        b5 = nn.Sequential(*blk5)

        # Each MRI image is (512,512) size.
        self.cnn_layer = nn.Sequential(
            # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            # MaxPool2D(kernel_size, stride, padding)
            b1,
            b2,
            b3,
            b4,
            b5,
            nn.AdaptiveAvgPool2d((1, 1)),

        )

        if is_mul:
            self.fully_conn = nn.Sequential(
                nn.Linear(2048, 4)
            )
        else:
            self.fully_conn = nn.Sequential(
                nn.Linear(2048, 2)
            )

    def forward(self, x):
        x = self.cnn_layer(x)
        x = x.flatten(1)
        x = self.fully_conn(x)
        # return x
        return F.log_softmax(x, dim=1)


# ----------------------------------------------------------------------------------------------------------------------
# Training and Validation process.
def train_valid_model(train_loader, valid_loader, epoch_num, is_mul):
    # Since tensorboard's add_scalar function doesn't work with torch on GPU,
    # I changed to use list to record the accuracy and loss vs each epoch.
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    # Set x axis of plot showing with integer.
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    # ------------------------------------------------------------------------------------------------------------------
    # Training process.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create Tensorboard writer
    # writer = SummaryWriter()
    # print("Tensorboard summary writer created.")

    # Initialize model and put it on cpu.
    model = CNN(is_mul=is_mul)
    model.to(device)
    print(model)

    # Total params of the model
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters of the model: " + str(total_params))
    # Total trainable params of the model
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters of the model: " + str(total_trainable_params))

    # Use cross entropy as the loss function.
    criterion = nn.CrossEntropyLoss()

    # Use Adam as optimizer. Manually set learning rate and fine tune it with experiments.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # multiply LR by 1 / 10 after every 100 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    print('LR Scheduler created')

    # Set the epoch numbers
    epoch_num = epoch_num

    # Set a seed to store the states files of model.
    seed = torch.initial_seed()
    print('Use seed : {}'.format(seed))
    os.makedirs("model_states_tmp/seed{}".format(seed), exist_ok=True)

    for epoch in range(epoch_num):
        # Model turns to train mode.
        model.train()

        train_loss = []
        train_accu = []

        for batch in tqdm(train_loader):
            # Each batch in torch consists of data and labels.
            data, labels = batch

            # Save some image samples in training process.
            # if (epoch % 100 == 0):
            #     grid = torchvision.utils.make_grid(data)
            #     writer.add_image("images", grid, epoch)

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

        lr_scheduler.step()

        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_train_accu = sum(train_accu) / len(train_accu)

        # Add train loss and accuracy into lists.
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_accu)

        # writer.add_scalar('train loss', avg_train_loss, epoch + 1)
        # writer.add_scalar('train accuracy', avg_train_accu, epoch + 1)

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

        # Add valid loss and accuracy into lists.
        valid_loss_list.append(avg_valid_loss)
        valid_acc_list.append(avg_valid_accu)

        # writer.add_scalar('valid loss', avg_valid_loss, epoch + 1)
        # writer.add_scalar('valid accuracy', avg_valid_accu, epoch + 1)

        print(f"[ Valid | {(epoch + 1) : 05d}] loss = {avg_valid_loss:.6g}, accu = {avg_valid_accu:.5f}")

        # Save the model states

        state = {
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'seed': seed
        }

        model_states_path = os.path.join("model_states_tmp/seed{}".format(seed),
                                         "model_states_epoch{}".format(epoch + 1))
        torch.save(state, model_states_path)
    # writer.close()

    # Save the records
    os.makedirs('tmp/resultsMatrix/seed{}'.format(seed), exist_ok=True)
    np.save('tmp/resultsMatrix/seed{}/train_acc.npy'.format(seed), np.array(train_acc_list))
    np.save('tmp/resultsMatrix/seed{}/valid_acc.npy'.format(seed), np.array(valid_acc_list))
    np.save('tmp/resultsMatrix/seed{}/train_loss.npy'.format(seed), np.array(train_loss_list))
    np.save('tmp/resultsMatrix/seed{}/valid_loss.npy'.format(seed), np.array(valid_loss_list))

    # Generate the plots.
    x1 = range(epoch_num)
    x2 = range(epoch_num)
    y1_1 = train_acc_list
    y1_2 = valid_acc_list
    y2_1 = train_loss_list
    y2_2 = valid_loss_list

    os.makedirs('tmp/imgs/seed{}'.format(seed), exist_ok=True)

    plt.plot(x1, y1_1, label='train accu', color='darkorange')
    plt.plot(x1, y1_2, label='valid accu', color='b')
    plt.title('Accuracy vs. epochs')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig("tmp/imgs/seed{}/".format(seed) + "accu.svg")
    plt.show()

    plt.plot(x2, y2_1, label='train loss', color='darkorange')
    plt.plot(x2, y2_2, label='valid loss', color='b')
    plt.title('Losses vs. epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss function value')
    plt.legend()

    # Don't put plt.savefig() behind plt.show()!
    plt.savefig("tmp/imgs/seed{}/".format(seed) + "loss.svg")
    plt.show()


if __name__ == "__main__":
    # Get params from command lines.
    p = argparse.ArgumentParser()
    p.add_argument("--isMul", default=False, action="store_true")
    p.add_argument("--epochNum", default=200, type=int)
    args = p.parse_args()

    # --------------------------------------------------------------------------------------------------

    # Choose to train for binary task or multi-class task.
    is_mul = args.isMul
    stf_K_fold = StratifiedKFold(n_splits=5)
    X, Y = PreProcessing.gen_X_Y(is_mul=is_mul)
    x_train, x_valid, y_train, y_valid = [], [], [], []
    # Acquire the train and valid sets.
    for train_idx, valid_idx in stf_K_fold.split(X, Y):
        # print("TRAIN:", train_idx, "TEST:", valid_idx)
        x_train, x_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = Y[train_idx], Y[valid_idx]
        break

    # Data augmentation.
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor(),
        transforms.Normalize(0.188, 0.198)
    ])

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(0.188, 0.198)
    ])

    # Convert into torch data loaders.
    torch_train_data = GetTorchData(x_train.reshape(2400, 512, 512), y_train, train_transform)
    torch_valid_data = GetTorchData(x_valid.reshape(600, 512, 512), y_valid, valid_transform)

    # Set batch size which will be used in training, validation and testing.
    batch_size = 100

    torch_train_loader = GetTorchDataLoader(torch_train_data, batch_size)
    torch_valid_loader = GetTorchDataLoader(torch_valid_data, batch_size)
    epoch_num = args.epochNum

    train_valid_model(torch_train_loader, torch_valid_loader, epoch_num, is_mul=is_mul)

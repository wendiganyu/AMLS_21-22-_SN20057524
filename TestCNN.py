import argparse

import torch

import CNN_Residual_Structure
from tqdm.auto import tqdm  # Progress bar

import PreProcessing
import torchvision.transforms as transforms
import torch.nn as nn


def CNN_Test(PATH, is_mul):
    """
    Measure the CNN model's performance on test set with saved model state file.

    Input
        PATH: The file path of saved model state file.
        is_mul: Load the model differently depending whether it will be used in binary or multiple
                classification tasks.
    """
    x_test, y_test = PreProcessing.gen_test_X_Y(is_mul=is_mul)
    batch_size = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(0.188, 0.198)
    ])
    criterion = nn.CrossEntropyLoss()

    model = CNN_Residual_Structure.CNN(is_mul=is_mul)

    checkpoint = torch.load(PATH, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['model'])

    # Model turns to evaluate mode.
    model.eval()

    test_loss = []
    test_accu = []
    torch_test_data = CNN_Residual_Structure.GetTorchData(x_test.reshape(200, 512, 512), y_test, test_transform)

    torch_test_loader = CNN_Residual_Structure.GetTorchDataLoader(torch_test_data, batch_size)
    for batch in tqdm(torch_test_loader):
        data, labels = batch
        # No back propagation in valid process.
        # Use torch.no_grad() to avoid using gradient.
        with torch.no_grad():
            res = model(data.float().to(device))

        loss = criterion(res, labels.to(device))

        accu = (res.argmax(dim=-1) == labels.to(device)).float().mean()

        test_loss.append(loss.item())
        test_accu.append(accu)

    avg_test_loss = sum(test_loss) / len(test_loss)
    avg_test_accu = sum(test_accu) / len(test_accu)

    print(f"[ Test ] loss = {avg_test_loss:.6g}, accu = {avg_test_accu:.5f}")


if __name__ == '__main__':
    # Get params from command lines.
    p = argparse.ArgumentParser()
    p.add_argument("--isMul", default=False, action="store_true")
    p.add_argument("--PATH", default="Please input the path of model state file.")
    args = p.parse_args()

    # ----------------------------------------------------------------------------------------------------
    is_mul = args.isMul

    PATH = args.PATH
    CNN_Test(PATH, is_mul)

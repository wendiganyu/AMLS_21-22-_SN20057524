import torch

import CNN_Residual_Structure
from tqdm.auto import tqdm  # Progress bar

import PreProcessing
import torchvision.transforms as transforms
import torch.nn as nn


def CNN_Test(PATH, is_mul):
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

    valid_loss = []
    valid_accu = []
    torch_valid_data = CNN_Residual_Structure.GetTorchData(x_test.reshape(200, 512, 512), y_test, test_transform)

    torch_valid_loader = CNN_Residual_Structure.GetTorchDataLoader(torch_valid_data, batch_size)
    for batch in tqdm(torch_valid_loader):
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

    print(f"[ Test ] loss = {avg_valid_loss:.6g}, accu = {avg_valid_accu:.5f}")


if __name__ == '__main__':
    is_mul = True

    for i in [1,2,3,4,5,6,7,8,9]:
        PATH_Mul = "model_states_tmp/Multiple/model_states_epoch19"+str(i)
        CNN_Test(PATH_Mul, is_mul)

    PATH_Mul = "model_states_tmp/Multiple/model_states_epoch200"
    CNN_Test(PATH_Mul, is_mul)

    # PATH_Binary = "model_states_tmp/Binary/model_states_epoch200"
    # is_mul = False
    # CNN_Test(PATH_Binary, is_mul)

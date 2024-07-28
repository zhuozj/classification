import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model import build_model
from utils import save_model, save_plots
from datasets import train_loader, valid_loader, dataset
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-e', "--epochs", type=int, default=20, help="number of epochs to train our network for")
parser.add_argument('-r', "--resume", action="store_true", help="resume model trainning")
args = vars(parser.parse_args())

# learning paramerts
lr = 0.001
epochs = args["epochs"]
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computation device: {device}")

# build model
model = build_model(
    pretrained=True, fine_tune=False, num_classes=len(dataset.classes)
).to(device)

# total parameters and trainable parameters
# total_params = sum(p.numel() for p in model.parameters())
# print(f"{total_params} total parameters.")

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# loss function
criterion = nn.CrossEntropyLoss()

if args["resume"]:
    print("resume training model")
    checkpoint = torch.load("./outputs/model.pth")
    epoch = checkpoint["epoch"]
    model_state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    criterion = checkpoint["loss"]

    model.load_state_dict(model_state_dict)

def train(model, trainloader, optimizer, criterion):
    model.train()
    print("Training")
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        loss.backward()
        optimizer.step()
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

def validate(model, testloader, criterion, class_names):
    model.eval()
    print("Validation")
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            outputs = model(image)

            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

            correct = (preds == labels).squeeze()
            for i in range(len(preds)):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))

    print('\n')
    for i in range(len(class_names)):
        print(f"Accuracy of class {class_names[i]}: {100 * class_correct[i] / class_total[i]:.2f}%")
    print('\n')

    return epoch_loss, epoch_acc

train_loss, valid_loss = [], []
train_acc, valid_acc = [], []

# start trainning
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_loader,
                                        optimizer, criterion)
    valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,
                                        criterion, dataset.classes)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)

    print(f"Training loss: {train_epoch_loss:.3f}, \
            Training acc: {train_epoch_acc:.2f}%")
    print(f"Validation loss: {valid_epoch_loss:.3f}, \
            Validation acc: {valid_epoch_acc:.2f}%")
    print('-'*50)

save_model(epochs, model, optimizer, criterion)

save_plots(train_acc, valid_acc, train_loss, valid_loss)
print("training complete")


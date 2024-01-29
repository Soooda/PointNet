import torch
from torch.utils.data import DataLoader
import os

from model.pointnet import PointNet
from data.dataset import Dataset

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

num_epochs = 200
batch_size = 32
learning_rate = 0.001
pin_memory = True if device.type == 'cuda' else False

train_ds = Dataset(os.path.abspath("datasets/"), num_points=1024, split="train", random_jitter=True, random_rotate=True, random_translate=True)
train_loader = DataLoader(train_ds, shuffle=True, num_workers=4, batch_size=batch_size, pin_memory=pin_memory)
valid_ds = Dataset(os.path.abspath("datasets/"), num_points=1024, split="test")
valid_loader = DataLoader(valid_ds, num_workers=4, batch_size=batch_size * 2, pin_memory=pin_memory)
print('Train dataset size: ', len(train_ds))
print('Valid dataset size: ', len(valid_ds))

model = PointNet(classes=40).to(device)
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=20, gamma=0.5)

def loss_fn(output, labels, m3x3, m64x64, alpha=0.001):
    criterion = torch.nn.NLLLoss()
    bs = output.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1).to(output.get_device())
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1).to(output.get_device())
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
    return criterion(output, labels) + alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)

for epoch in range(1, num_epochs + 1):
    checkpoint = os.sep.join(("checkpoints", str(epoch) + ".pth"))
    if os.path.exists(checkpoint):
        if os.path.exists(os.sep.join(("checkpoints", str(epoch + 1) + ".pth"))):
            continue
        temp = torch.load(checkpoint)
        model.load_state_dict(temp["model"])
        optim.load_state_dict(temp["optimizer"])
        scheduler.load_state_dict(temp["scheduler"])
        continue

    running_loss = 0.0
    model.train()

    for i, data in enumerate(train_loader, 0):
        inputs, labels, _, _ = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels.squeeze()
        optim.zero_grad()
        outputs, m3x3, m64x64 = model(inputs.transpose(1, 2))
        loss = loss_fn(outputs, labels, m3x3, m64x64)
        loss.backward()
        optim.step()

        running_loss += loss.item()

    model.eval()
    correct = total = 0
    if valid_loader:
        with torch.no_grad():
            for data in valid_loader:
                inputs, labels, _, _ = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.squeeze()
                outputs, _, _ = model(inputs.transpose(1, 2))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100. * correct / total
    # print statistics
    print("Epoch {:<4} Loss: {:<8.4f} Valid accuracy: {:<6.2f}%".format(epoch, running_loss, val_acc))

    scheduler.step() 
    # Checkpoints
    checkpoints = {
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "scheduler": optim.state_dict(),
    }
    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")
    torch.save(checkpoints, os.sep.join(("checkpoints", str(epoch) + ".pth")))
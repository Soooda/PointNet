import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from sklearn.metrics import confusion_matrix
import itertools

from model.pointnet import PointNet
from data.dataset import Dataset

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

with open(os.path.join("datasets", "modelnet40_hdf5_2048", "shape_names.txt"), 'r') as f:
    label_map = f.read().splitlines()

checkpoint = os.path.join("checkpoints", "200.pth")
model = PointNet(classes=40).to(device)
temp = torch.load(checkpoint, map_location=device)
model.load_state_dict(temp["model"])
d = Dataset(os.path.abspath("datasets/"), num_points=1024, split="test")
valid_loader = DataLoader(d, batch_size=64)
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for i, data in enumerate(valid_loader):
        inputs, labels, _, _ = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels.squeeze()
        outputs, _, _ = model(inputs.transpose(1, 2))
        _, predicted = torch.max(outputs.data, 1)
        all_preds += list(predicted.cpu().numpy())
        all_labels += list(labels.cpu().numpy())

        print("Batch [{:>4d} / {:>4d}]\r".format(i + 1, len(valid_loader)), end="")

print()

cm = confusion_matrix(all_labels, all_preds)

# function from https://deeplizard.com/learn/video/0LhiS6yu2qQ
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plt.figure(figsize=(9, 9))
plot_confusion_matrix(cm, label_map, normalize=False)

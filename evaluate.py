import torch
import os
import open3d as o3d
import random

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
class_choice = None

model = PointNet(classes=40).to(device)
temp = torch.load(checkpoint)
model.load_state_dict(temp["model"])
d = Dataset(os.path.abspath("datasets/"), num_points=1024, split="test", class_choice=class_choice)
model.eval()
# print("datasize:", d.__len__())

item = random.randint(0, len(d))
ps, lb, n, f = d[item]
print(ps.size(), f)

# Visualises the point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(ps.numpy())
o3d.visualization.draw_geometries([pcd], width=800, height=600)

ps = ps.to(device)
outputs, _, _ = model(ps.unsqueeze(0).transpose(1, 2))
_, predicted = torch.max(outputs.data, 1)
print("Ground Truth: {:15s} Predicted: {:15s}".format(label_map[lb[0]], label_map[predicted[0]]))
# PointNet
The coding for my final presentation of Tokyo Institute of Technology's ART.T466 3D Computer Vision

## Environment

```bash
$ conda create -n PointNet pip python=3.10
# Nvidia Cuda
$ conda install pytorch==1.12.1 cudatoolkit=11.3 -c pytorch
# Mac MPS
$ conda install pytorch::pytorch -c pytorch

$ pip install h5py open3d
```

## Dataset

The ModelNet40 dataset and the dataset loader are attained from [PointCloudDatasets](https://github.com/antao97/PointCloudDatasets).

Put the extracted folder `modelnet40_hdf5_2048` under `pointnet/datasets` directory.

## References

* [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)

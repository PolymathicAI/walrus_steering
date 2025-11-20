import argparse
import glob
import os

import h5py as h5
import numpy as np


def fourier_downsample(x, factor=1 / 8):
    """Does resampling in Fourier space"""
    H, W, D = x.shape[2:5]
    h, w, d = int(H * factor), int(W * factor), int(D * factor)
    xfft = np.fft.rfftn(x, axes=(2, 3, 4), norm="forward")
    xfft = np.fft.fftshift(xfft, axes=(2, 3))
    xfft = xfft[:, :, :, :, : d // 2 + 1]
    xfft = xfft[:, :, :, W // 2 - w // 2 : W // 2 + w // 2, :]
    xfft = xfft[:, :, H // 2 - h // 2 : H // 2 + h // 2, :, :]
    xfft = np.fft.ifftshift(xfft, axes=(2, 3))
    out = np.fft.irfftn(xfft, axes=(2, 3, 4), norm="forward")
    return out


def copy_attrs(in_file, out_file):
    for key in in_file.attrs.keys():
        out_file.attrs[key] = in_file.attrs[key]


def copy_and_downsample_dims(in_file, out_file, factor=0.25):
    old_dims = in_file["dimensions"]
    new_dims = out_file.create_group("dimensions")
    copy_attrs(old_dims, new_dims)
    for key in old_dims.keys():
        if key != "time":
            shape = old_dims[key].shape
            left = old_dims[key][0]
            right = old_dims[key][-1]
            out_seq = np.linspace(left, right, int(shape[0] * factor))
            ds = new_dims.create_dataset(key, data=out_seq)
            copy_attrs(old_dims[key], ds)
        else:
            new_dims.create_dataset(key, data=old_dims[key])
            copy_attrs(old_dims[key], new_dims[key])


def copy_and_downsample_bc(in_file, out_file, factor=0.25):
    in_file = in_file["boundary_conditions"]
    out_file = out_file.create_group("boundary_conditions")
    copy_attrs(in_file, out_file)
    for key in in_file.keys():
        newf = out_file.create_group(key)
        copy_attrs(in_file[key], newf)
        # for subkey in in_file[key].keys():
        old_mask = in_file[key]["mask"]
        old_shape = old_mask.shape
        new_mask = np.zeros(int(old_shape[0] * factor), dtype=bool)
        newf.create_dataset("mask", data=new_mask)
        copy_attrs(in_file[key]["mask"], newf["mask"])


def copy_and_downsample_field(in_file, out_file, factor=0.25):
    field_types = ["t0_fields", "t1_fields", "t2_fields"]
    for field_type in field_types:
        in_field = in_file[field_type]
        out_field = out_file.create_group(field_type)
        copy_attrs(in_field, out_field)
        for key in in_field.keys():
            old_data = in_field[key]
            new_data = fourier_downsample(old_data, factor)
            ds = out_field.create_dataset(key, data=new_data, dtype="f4")
            copy_attrs(in_field[key], ds)


def copy_scalars(in_file, out_file):
    in_file = in_file["scalars"]
    out_file = out_file.create_group("scalars")
    copy_attrs(in_file, out_file)
    for key in in_file.keys():
        newf = out_file.create_dataset(key, data=in_file[key])
        copy_attrs(in_file[key], newf)


def copy_and_downsample(in_file, out_file, factor=0.25):
    copy_attrs(in_file, out_file)
    copy_and_downsample_dims(in_file, out_file, factor)
    copy_and_downsample_bc(in_file, out_file, factor)
    copy_and_downsample_field(in_file, out_file, factor)
    copy_scalars(in_file, out_file)


if __name__ == "__main__":
    # Get an index from the argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    index = args.index

    source = "/mnt/home/polymathic/ceph/the_well/datasets/MHD/data"
    target = "/mnt/home/polymathic/ceph/the_well/datasets/MHD_64/data"

    subdirs = ["train", "valid", "test"]

    file_pairs = []
    for subdir in subdirs:
        files = glob.glob(f"{source}/{subdir}/*.hdf5")
        os.makedirs(f"{target}/{subdir}", exist_ok=True)
        for source_file in files:
            print(source_file)
            tar_file = source_file.replace(source, target)
            file_pairs.append((source_file, tar_file))

    source_file, tar_file = file_pairs[index]
    with h5.File(source_file, "r") as source_file:
        with h5.File(tar_file, "w") as target_file:
            copy_and_downsample(source_file, target_file, factor=0.25)

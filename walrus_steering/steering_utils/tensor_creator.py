#### ––– Injection Tensor Creator ––– ####
import pickle
import torch


def save_tensor(tensor, fname, path="experiments/activations/"):
    with open(path+fname, 'wb') as handle:
        pickle.dump(tensor, handle)

def normalise_tensor(activation_tensor, dims=(3,4,5)): # (0,1,3,4,5)
    # For tensor shape: [6, 1, 1408, 31, 32, 1]
    #                   [T, B,   C,   H,  W, D] (correct?) 
    mean = activation_tensor.mean(dim=dims, keepdim=True)
    std = activation_tensor.std(dim=dims, keepdim=True)
    return (activation_tensor - mean) / std

def read_files(file_list):
    tensor_list = []
    for file_path in file_list:
        with open("experiments/activations/"+file_path+".pickle", 'rb') as handle:
            tensor_dict = pickle.load(handle)
            tensor = tensor_dict["blocks.39"]  # Original shape [6, 1, 1408, 31, 32, 1]
            normalized_tensor = normalise_tensor(tensor)
            tensor_list.append(normalized_tensor)
    return tensor_list

def average_tensors(tensor_list):
    if len(tensor_list) > 1:
        return torch.stack(tensor_list).mean(dim=0)
    else:
        return tensor_list[0]

laminar_files_fast = [
                    "shear_flow_Reynolds_1e4_Schmidt_1e0(double-v)[FullRes][dt_stride=2,v_factor=2]",
                    "shear_flow_Reynolds_1e4_Schmidt_1e1(double-v)[FullRes][dt_stride=2,v_factor=2]",
                    "shear_flow_Reynolds_1e5_Schmidt_1e1(double-v)[FullRes][dt_stride=2,v_factor=2]",
                    "shear_flow_Reynolds_1e5_Schmidt_2e-1(double-v)[FullRes][dt_stride=2,v_factor=2]",
                    "shear_flow_Reynolds_1e5_Schmidt_5e0(double-v)[FullRes][dt_stride=2,v_factor=2]",
                    "shear_flow_Reynolds_5e4_Schmidt_2e-1(double-v)[FullRes][dt_stride=2,v_factor=2]",
                    "shear_flow_Reynolds_5e5_Schmidt_1e-1(double-v)[FullRes][dt_stride=2,v_factor=2]",
                    "shear_flow_Reynolds_5e5_Schmidt_1e1(double-v)[FullRes][dt_stride=2,v_factor=2]",
                    "shear_flow_Reynolds_5e5_Schmidt_2e0(double-v)[FullRes][dt_stride=2,v_factor=2]",
                    "shear_flow_Reynolds_5e5_Schmidt_5e-1(double-v)[FullRes][dt_stride=2,v_factor=2]"
                    ]


laminar_files_slow = [
                    "shear_flow_Reynolds_1e4_Schmidt_1e0[FullRes][dt_stride=1]",
                    "shear_flow_Reynolds_1e4_Schmidt_1e1[FullRes][dt_stride=1]",
                    "shear_flow_Reynolds_1e5_Schmidt_1e1[FullRes][dt_stride=1]",
                    "shear_flow_Reynolds_1e5_Schmidt_2e-1[FullRes][dt_stride=1]",
                    "shear_flow_Reynolds_1e5_Schmidt_5e0[FullRes][dt_stride=1]",
                    "shear_flow_Reynolds_5e4_Schmidt_2e-1[FullRes][dt_stride=1]",
                    "shear_flow_Reynolds_5e5_Schmidt_1e-1[FullRes][dt_stride=1]",
                    "shear_flow_Reynolds_5e5_Schmidt_1e1[FullRes][dt_stride=1]",
                    "shear_flow_Reynolds_5e5_Schmidt_2e0[FullRes][dt_stride=1]",
                    "shear_flow_Reynolds_5e5_Schmidt_5e-1[FullRes][dt_stride=1]"
                    ]


feature_tensors = read_files(laminar_files_fast)
control_tensors = read_files(laminar_files_slow)

# Average over each file, separately for each batch, hence each avg tensor is really 6 tensors
avg_feature_tensor = average_tensors(feature_tensors)
avg_control_tensor = average_tensors(control_tensors)

# Average over the batches and channels
injection_tensor = avg_feature_tensor.mean((0,1), True) - avg_control_tensor.mean((0,1), True)

save_tensor(injection_tensor,"shear_flow_laminar:[dt_stride=2,v_factor=2]-[dt_stride=1,v_factor=1].pickle")


# --- FILES --- #
# vortex_files_full_set = [
#                     "shear_flow_Reynolds_1e4_Schmidt_1e-1",
#                     "shear_flow_Reynolds_1e4_Schmidt_2e-1",
#                     "shear_flow_Reynolds_1e4_Schmidt_2e0",
#                     "shear_flow_Reynolds_1e4_Schmidt_5e-1",
#                     "shear_flow_Reynolds_1e4_Schmidt_5e0",
#                     "shear_flow_Reynolds_1e5_Schmidt_1e-1",
#                     "shear_flow_Reynolds_1e5_Schmidt_1e0",
#                     "shear_flow_Reynolds_1e5_Schmidt_2e0",
#                     "shear_flow_Reynolds_1e5_Schmidt_5e-1",
#                     "shear_flow_Reynolds_5e4_Schmidt_1e-1",
#                     "shear_flow_Reynolds_5e4_Schmidt_1e0",
#                     "shear_flow_Reynolds_5e4_Schmidt_1e1",
#                     "shear_flow_Reynolds_5e4_Schmidt_2e0",
#                     "shear_flow_Reynolds_5e4_Schmidt_5e-1",
#                     "shear_flow_Reynolds_5e4_Schmidt_5e0",
#                     "shear_flow_Reynolds_5e5_Schmidt_1e0",
#                     "shear_flow_Reynolds_5e5_Schmidt_2e-1",
#                     "shear_flow_Reynolds_5e5_Schmidt_5e0"
#                     ]

# laminar_files_full_set = [
#                     "shear_flow_Reynolds_1e4_Schmidt_1e0",
#                     "shear_flow_Reynolds_1e4_Schmidt_1e1",
#                     "shear_flow_Reynolds_1e5_Schmidt_1e1",
#                     "shear_flow_Reynolds_1e5_Schmidt_2e-1",
#                     "shear_flow_Reynolds_1e5_Schmidt_5e0",
#                     "shear_flow_Reynolds_5e4_Schmidt_2e-1",
#                     "shear_flow_Reynolds_5e5_Schmidt_1e-1",
#                     "shear_flow_Reynolds_5e5_Schmidt_1e1",
#                     "shear_flow_Reynolds_5e5_Schmidt_2e0",
#                     "shear_flow_Reynolds_5e5_Schmidt_5e-1"
#                     ]


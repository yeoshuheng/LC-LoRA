import numpy as np
import pathos.multiprocessing as pmp
import torch, zlib
import src.compression.LowRankLinear as LowRankLinear

def decode_data(checkpoint):
    """
    @param checkpoint : GZIP Encoded checkpoint

    @return : Decoded checkpoint.
    """
    return np.frombuffer(zlib.decompress(checkpoint), dtype = np.float32)

def restoreLinearLayer(alpha, beta, s1, s2, base):
    """
    @param alpha : Left component of the decomposition.
    @param beta : Right component of the decomposition.

    @return The converted weights of the original model according to the decomposition.
    """
    return torch.add(torch.add(base, torch.matmul(alpha, beta)), torch.matmul(s1, s2))

def restore_state_dict(decoded_checkpoint, decoded_decomp_checkpoint, bias, base_dict, rank, org, decomposed_layers):
    """
    @param decoded_checkpoint: The decoded checkpoint of normal weights from zlib.
    @param decoded_decomp_checkpoint: The decoded checkpoint of decomposed weights from zlib.
    @param bias : The bias dictionary of the model.
    @param base_dict : The base dictionary of the model which helps us understand its structure.
    @param rank : The rank of the decomposition used for the linear layers.
    @param org : The original model state dictionary, when the branch was first taken.
    @param decomposed_layers : list of layers that have undergone decomposition. 

    @return Restored state_dict.
    """
    last_idx, last_idx_dcomp = 0, 0
    for layer_name, init_tensor in base_dict.items():
        if "bias" in layer_name:
            base_dict[layer_name] = bias[layer_name]
            continue
        dim = init_tensor.numpy().shape
        if not dim:
            continue
        if layer_name in decomposed_layers: # Restoration procedure for dense layers.
            if rank == -1:
                rr = min(dim[0], dim[1]) // 4
                t_element_alpha = dim[0] * rr
                t_element_beta = dim[1] * rr
            else:
                t_element_alpha = dim[0] * rank
                t_element_beta = dim[1] * rank
            alpha = decoded_decomp_checkpoint[last_idx_dcomp : last_idx_dcomp + t_element_alpha]
            last_idx_dcomp += t_element_alpha
            beta = decoded_decomp_checkpoint[last_idx_dcomp : last_idx_dcomp + t_element_beta]
            last_idx_dcomp += t_element_beta
            sparse1 = decoded_decomp_checkpoint[last_idx_dcomp : last_idx_dcomp + t_element_alpha]
            last_idx_dcomp += t_element_alpha
            sparse2 = decoded_decomp_checkpoint[last_idx_dcomp : last_idx_dcomp + t_element_beta]
            last_idx_dcomp += t_element_beta
            alpha = torch.unflatten(torch.from_numpy(np.copy(alpha)), -1, (dim[0], rank))
            beta = torch.unflatten(torch.from_numpy(np.copy(beta)), -1, (rank, dim[1]))
            sparse1 = torch.unflatten(torch.from_numpy(np.copy(sparse1)), -1, (dim[0], rank))
            sparse2 = torch.unflatten(torch.from_numpy(np.copy(sparse2)), -1, (rank, dim[1]))
            restored_decomp = restoreLinearLayer(alpha, beta, sparse1, sparse2, org[layer_name])
            base_dict[layer_name] = restored_decomp
        elif "classifier" in layer_name:
            base_dict[layer_name] = bias[layer_name]
        else: # Restoration procedure for convolutional layers.
            t_elements = np.prod(dim)
            needed_ele = decoded_checkpoint[last_idx : last_idx + t_elements]
            base_dict[layer_name] = torch.unflatten(torch.from_numpy(np.copy(needed_ele)), -1, dim)
            last_idx += t_elements
    return base_dict
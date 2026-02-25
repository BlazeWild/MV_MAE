import os
import torch
import torch.nn.functional as F 

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from pretraining.data.dataset import UAVHumanDataset


def main():
    base_dir = os.path.dirname(__file__)
    data_root = os.path.abspath(os.path.join(base_dir, '../../datasets/UAVHuman_480p_mp4/Action_Videos'))
    train_split = os.path.abspath(os.path.join(base_dir, '../../datasets/UAVHuman_480p_mp4/train_split.txt'))
    codebook_path = os.path.abspath(os.path.join(base_dir, '../codebook_ckpt/mv_codebook_1024.pt'))

    print("Loading Codebook")
    codebook = torch.load(codebook_path)
    print("Codebook loaded successfully")
    print("Codebook shape:", codebook.shape)
    print("Codebook:", codebook)

    print("Loading a sample video")
    dataset = UAVHumanDataset(
        data_root=data_root, 
        split_file=train_split, 
        num_segments=8, 
        gop_size=16,
        is_train=False
    )

    _,mvs, label = dataset[0]
    print("MVs shape:", mvs.shape)
    print("Label:", label)

    mvs_permuted = mvs.permute(0,2,3,4,1)
    mvs_flat = mvs_permuted.reshape(-1,2)


    print(f"Extracted {mvs_flat.shape[0]} motion vectors")

    distances = torch.cdist(mvs_flat, codebook)
    token_ids = torch.argmin(distances, dim=1)
    reconstructed_mvs = codebook[token_ids]

    print("Results and sanity check")
    unique_tokens = len(torch.unique(token_ids))
    print(f"Unique tokens used: {unique_tokens}/{codebook.shape[0]}")

    zero_tokens = (token_ids ==0).sum().item()
    zero_percentage = (zero_tokens / len(token_ids)) * 100
    print(f"Zero tokens: {zero_tokens} ({zero_percentage:.2f}%)")

    mse_loss = F.mse_loss(reconstructed_mvs, mvs_flat).item()
    print(f"Mean Squared Error (Reconstruction Loss): {mse_loss:.4f}")


if __name__ == "__main__":
    main()

    

    
    
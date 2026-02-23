import torch
state_dict = torch.load('model_zoo/video_mae/vit_s_k400_ft.pth', map_location='cpu')
if 'model' in state_dict:
    state_dict = state_dict['model']
print(f"Num keys: {len(state_dict)}")
for k in list(state_dict.keys())[:10]:
    print(f"{k}: {state_dict[k].shape}")
for k in list(state_dict.keys())[-10:]:
    print(f"{k}: {state_dict[k].shape}")

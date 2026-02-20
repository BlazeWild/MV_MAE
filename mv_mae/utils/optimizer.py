import torch.optim as optim

def get_optimizer(model, learning_rate=1e-4, weight_decay=0.05):
    """
    Creates an AdamW optimizer tailored fofr transfromer
    """
    decay = []
    no_decay=[]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape)==1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    
    optim_groups = [
        {"params":decay, "weight_decay":weight_decay},
        {"params":no_decay, "weight_decay":0.0}
    ]
    
    optimizer = optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer
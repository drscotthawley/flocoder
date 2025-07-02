# flocoder/ot.py - OT pairing implementation following StyleGuide.md

import torch, numpy as np

def compute_ot_pairing_vanilla(source, target, reg=0.1):
    """Original slow but robust OT using POT library"""
    try: import ot
    except ImportError: raise ImportError("POT library required: pip install POT")
    
    B = source.shape[0]
    source_flat = source.view(B, -1).detach().cpu().numpy()  # flatten for distance computation
    target_flat = target.view(B, -1).detach().cpu().numpy()
    a, b = np.ones(B) / B, np.ones(B) / B  # uniform distributions
    M = ot.dist(source_flat, target_flat, metric='sqeuclidean')  # cost matrix
    P = ot.sinkhorn(a, b, M, reg=reg)  # solve OT with Sinkhorn regularization
    
    ot_indices, used_targets = [], set()  # convert transport plan to discrete pairing
    for i in range(B):
        available_targets = [j for j in range(B) if j not in used_targets]
        if available_targets:
            best_target = max(available_targets, key=lambda j: P[i, j])  # pick target with highest transport probability
            ot_indices.append(best_target)
            used_targets.add(best_target)
        else: ot_indices.append(i)  # fallback if no targets left
    
    return torch.tensor(ot_indices, dtype=torch.long)

def compute_ot_pairing_torchcfm(source, target, method='sinkhorn', reg=0.05, debug=False):
    """Fast OT using torchcfm library with debug output"""
    try:
        from torchcfm.optimal_transport import OTPlanSampler
        B = source.shape[0]
        source_flat = source.view(B, -1).detach()
        target_flat = target.view(B, -1).detach()
        ot_sampler = OTPlanSampler(method=method, reg=reg)  # initialize OT sampler
        result = ot_sampler.sample_plan(source_flat, target_flat)  # get transport plan
        
        if debug:  # inspect what we actually got
            print(f"DEBUG: result type = {type(result)}")
            print(f"DEBUG: result = {result}")
            if hasattr(result, 'shape'): print(f"DEBUG: result.shape = {result.shape}")
            if hasattr(result, 'dtype'): print(f"DEBUG: result.dtype = {result.dtype}")
            if isinstance(result, (list, tuple)):
                print(f"DEBUG: result length = {len(result)}")
                if len(result) > 0: print(f"DEBUG: first element = {result[0]}, type = {type(result[0])}")
        
        if hasattr(result, 'shape') and len(result.shape) == 2:  # coupling matrix [B, B]
            indices = result.argmax(dim=1)  # greedy assignment
        elif hasattr(result, 'long'): indices = result.long()  # already tensor-like
        else: indices = torch.tensor(result, dtype=torch.long)  # assume list/array
        
        indices = indices.long()[:B]  # ensure right shape and type
        if debug:
            print(f"DEBUG: final indices = {indices}")
            print(f"DEBUG: final indices.shape = {indices.shape}")
            print(f"DEBUG: final indices.dtype = {indices.dtype}")
        return indices
        
    except Exception as e:
        print(f"torchcfm OT failed: {e}, falling back to approximate method")
        return compute_ot_pairing_approximate(source, target)

def compute_ot_pairing_approximate(source, target):
    """Fast approximate OT using greedy distance matching"""
    B = source.shape[0]
    source_flat = source.view(B, -1)
    target_flat = target.view(B, -1)
    distances = torch.cdist(source_flat, target_flat)  # compute pairwise distances
    
    indices = torch.zeros(B, dtype=torch.long, device=source.device)  # greedy assignment
    used = torch.zeros(B, dtype=torch.bool, device=source.device)
    for i in range(B):
        masked_distances = distances[i].clone()
        masked_distances[used] = float('inf')  # find closest unused target
        best_j = masked_distances.argmin()
        indices[i] = best_j
        used[best_j] = True
    return indices

def compute_ot_pairing(source, target, debug=False):
    """Main wrapper - change this line to switch methods"""
    #return compute_ot_pairing_torchcfm(source, target, debug=debug)
    # return compute_ot_pairing_vanilla(source, target) 
    return compute_ot_pairing_approximate(source, target)

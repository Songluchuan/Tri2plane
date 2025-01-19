import torch

def Rotate_y_180(X, pos='right'):
    R = torch.eye(3).to(X.device)
    R[0,0] = -1.0
    R[2,2] = -1.0
    if pos == 'right':
        X = torch.matmul(X, R)
    else:
        X = torch.matmul(R, X)
    return X

def Rotate_z_180(X, pos='right'):
    R = torch.eye(3).to(X.device)
    R[0,0] = -1.0
    R[1,1] = -1.0
    if pos == 'right':
        X = torch.matmul(X, R)
    else:
        X = torch.matmul(R, X)
    return X
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cudnn:", torch.backends.cudnn.version())

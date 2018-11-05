import torch
import os

checkpoint_path = 'R50_FPN-test/model_0000250.pth'
assert os.path.exists(checkpoint_path)
checkpoint = torch.load(checkpoint_path)

model = checkpoint['model']
layers = {k for k in model.keys() if k.find('mask')>0}

for layer in layers:
    print(layer)
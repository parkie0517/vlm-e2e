import torch
import torch.nn as nn

class simpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = simpleMLP()
num_params = 0

for name, p in model.named_parameters():
    num_param = p.numel()
    num_params += num_param
    print(name, num_param)
print(num_params)
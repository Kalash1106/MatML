import torch.nn as nn

class ContrastiveNet(nn.Module):
    def __init__(self, base_model, hidden_dim = 128) -> None:
        super(ContrastiveNet, self).__init__()
        self.contnet = nn.Sequential(
            base_model,
            nn.ReLU(inplace=True),
            nn.Linear(1000, hidden_dim)
        )
    
    def forward(self, x):
        return self.contnet(x)
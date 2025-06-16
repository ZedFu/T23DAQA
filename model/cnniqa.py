import torch
from torch import nn 
import torch.nn.functional as F

class CNNIQAnet(nn.Module):
    def __init__(self, ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800):
        super(CNNIQAnet, self).__init__()
        self.conv1  = nn.Conv2d(3, n_kers, ker_size)
        self.fc1    = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2    = nn.Linear(n1_nodes, n2_nodes)
        self.fc3    = nn.Linear(n2_nodes, 3)
        self.dropout = nn.Dropout()

    def forward(self, x):
        N,C = x.shape[0], x.shape[1]
        x = x.unfold(2,32,32).unfold(3,32,32)
        x = x.contiguous().view(-1,C,32,32)

        h  = self.conv1(x)

        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h  = torch.cat((h1, h2), 1)  # max-min pooling
        h  = h.squeeze(3).squeeze(2)

        h  = F.relu(self.fc1(h))
        h  = self.dropout(h)
        h  = F.relu(self.fc2(h))

        q  = self.fc3(h)
        q = q.view(N,-1,3).mean(dim=1)
        return q

if __name__ == "__main__":
    input = torch.randn(4,3,512,512)
    output = input.unfold(2,32,32).unfold(3,32,32)
    output = output.contiguous().view(4, 3, -1, 32, 32)
    model = CNNIQAnet()
    print(model(input).shape)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size=3, hidden_size=3, output_size=3):
        super(MLP, self).__init__()

        ### YOUR CODE HERE
        # Simple 1-hidden-layer MLP
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        ### END YOUR CODE

        
    def forward(self, x):

        ### YOUR CODE HERE
        # x: [batch_size, input_size]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
        ### END YOUR CODE

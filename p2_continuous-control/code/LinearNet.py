import torch.nn as nn


class LinearNet(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(LinearNet, self).__init__()

        seq = self._create_layers(input_size, hidden_layers, output_size)
        self.model = nn.Sequential(*seq)
        self._initialize_weights()

    def _create_layers(self, input_size, hidden_layers, output_size):
        seq = [nn.Linear(input_size, hidden_layers[0])]
        for i in range(1, len(hidden_layers)):
            seq = seq + [nn.ReLU()]
            seq = seq + [nn.Linear(hidden_layers[i - 1], hidden_layers[i])]

        seq = seq + [nn.ReLU()]
        seq = seq + [nn.Linear(hidden_layers[-1], output_size)]
        return seq

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight = nn.init.xavier_uniform_(module.weight, gain=1)

    def forward(self, state):
        return self.model.forward(state)

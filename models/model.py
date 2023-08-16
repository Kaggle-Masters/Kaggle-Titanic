from torch import nn


class Baseline(nn.Module):
    """
    Base model with 
    """

    def __init__(self, in_features):
        super(Baseline, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.layers(x)

        return x.view(-1)


class Net1(nn.Module):
    """
    Regression model with 3 Layers + Sigmoid activation
    """

    def __init__(self, in_features):
        super(Net1, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.Sigmoid(),
            nn.Linear(32, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.layers(x)

        return x.view(-1)


class Net2(nn.Module):
    """
    Regression model with 3 Layers + ReLU activation
    """

    def __init__(self, in_features):
        super(Net2, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.layers(x)

        return x.view(-1)


class Net3(nn.Module):
    """"
    Regression model with 3 Layers + Sigmoid activation + Dropout 
    """

    def __init__(self, in_features):
        super(Net3, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.layers(x)

        return x.view(-1)


class Net4(nn.Module):
    """"
    Classification model with 3 Layers + Sigmoid activation + Dropout
    """

    def __init__(self, in_features):
        super(Net4, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.layers(x)
        
        return x

class Net5(nn.Module):
    """"
    Classification model with 3 Layers + Sigmoid activation + Dropout
    """

    def __init__(self, in_features):
        super(Net5, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.layers(x)
        
        return x
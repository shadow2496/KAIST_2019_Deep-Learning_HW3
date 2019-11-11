from torch import nn, optim


class dA(nn.Module):
    def __init__(self, in_features, out_features):
        super(dA, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_features, in_features),
            nn.Sigmoid()
        )

        self.decoder[0].weight.data.copy_(self.encoder[0].weight.data.transpose(0, 1))

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)


class SdA(nn.Module):
    def __init__(self, config):
        super(SdA, self).__init__()

        self.layers = []
        in_features = config.input_features
        for out_features in config.hidden_features:
            layer = dA(in_features, out_features)
            in_features = out_features
            self.layers.append(layer)
        self.layers.append(nn.Linear(in_features, config.classes))

        if config.is_train:
            self.bce_criterion = nn.BCELoss()
            self.ce_criterion = nn.CrossEntropyLoss()

            self.da_optimizers = []
            for layer in self.layers[:-1]:
                optimizer = optim.SGD(layer.parameters(), lr=config.lr,
                                      momentum=config.momentum, weight_decay=config.weight_decay)
                self.da_optimizers.append(optimizer)

            sda_params = []
            for layer in self.layers[:-1]:
                sda_params.extend(layer.encoder.parameters())
            sda_params.extend(self.layers[-1].parameters())
            self.sda_optimizer = optim.SGD(sda_params, lr=config.lr,
                                           momentum=config.momentum, weight_decay=config.weight_decay)

    def forward(self, x):
        h = x
        for layer in self.layers[:-1]:
            h = layer.encoder(h)
        return self.layers[-1](h)

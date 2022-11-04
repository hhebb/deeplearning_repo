class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding='same'),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(8, 16, 3, padding='same'),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(16, 8, 2, stride=2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(8, 1, 2, stride=2),
            torch.nn.BatchNorm2d(1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded

    def predict(self, x):
        pred = self.encoder(x)
        
        return pred
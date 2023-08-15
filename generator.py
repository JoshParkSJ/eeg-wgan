import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, batch_size, num_channels, sequence_length):
        super(Generator, self).__init__()
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.sequence_length = sequence_length

        self.start = nn.ConvTranspose1d(100, 50, kernel_size=9, stride=1, padding=4)
        self.end = nn.Linear(50 * batch_size, sequence_length)
        self.relu = nn.LeakyReLU(0.5)
        self.upsample = nn.Upsample(scale_factor=2)
        self.block = nn.Sequential(
            nn.ConvTranspose1d(50, 50, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(50, affine=True),
            nn.ReLU(),
            nn.ConvTranspose1d(50, 50, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(50, affine=True),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.start(x)
        x = self.relu(x)
        x = self.block(x)
        x = self.upsample(x)
        x = self.block(x)
        x = self.upsample(x)
        x = self.block(x)
        x = self.upsample(x)
        x = self.block(x)
        x = self.upsample(x)
        x = self.block(x)
        x = self.upsample(x)
        x = self.block(x)
        x = self.upsample(x)
        x = self.block(x)
        x = x.view(self.batch_size, -1) # flatten
        x = self.end(x)
        return x.reshape(self.batch_size, self.num_channels, self.sequence_length)

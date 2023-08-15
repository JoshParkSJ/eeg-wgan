import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, batch_size, num_channels, sequence_length):
        super(Discriminator, self).__init__()
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.sequence_length = sequence_length

        self.start = nn.Conv1d(1, 50, kernel_size=9, stride=1, padding=4)
        self.end = nn.Linear(50 * 11, 1)
        self.lrelu = nn.LeakyReLU(0.2)
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        self.block = nn.Sequential(
            nn.Conv1d(50, 50, kernel_size=9, stride=1, padding=4),
            nn.InstanceNorm1d(50, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv1d(50, 50, kernel_size=9, stride=1, padding=4),
            nn.InstanceNorm1d(50, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = x.float()
        x = self.start(x)
        x = self.lrelu(x)
        x = self.block(x)
        x = self.downsample(x)
        x = self.block(x)
        x = self.downsample(x)
        x = self.block(x)
        x = self.downsample(x)
        x = self.block(x)
        x = self.downsample(x)
        x = self.block(x)
        x = self.downsample(x)
        x = self.block(x)
        x = self.downsample(x)
        x = self.block(x)
        x = self.downsample(x)
        x = self.block(x)
        x = self.downsample(x)
        x = x.view(self.batch_size, -1) # flatten
        x = self.end(x)
        return x.reshape(self.batch_size)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import generator as Generator
import discriminator as Discriminator

LAMBDA_GP = 10

def train(training_data, batch_size, num_channels, sequence_length, num_epochs):
    print_interval = 50
    critic_err, gen_err, generated_data = [], [], None
    G = Generator(batch_size, num_channels, sequence_length)
    D = Discriminator(batch_size, num_channels, sequence_length)
    weights_init(G)
    weights_init(D)
    critic_optimizer = optim.RMSprop(D.parameters(), lr=2e-4)
    gen_optimizer = optim.RMSprop(G.parameters(), lr=2e-4)

    for epoch in range(num_epochs):
        for _ in range(5):
            real = training_data.clone().detach()
            real.requires_grad=True
            noise = sample_normal_distribution(batch_size, 100, 1)
            fake = G(noise)
            critic_real = D(real).reshape(-1) # flatten
            critic_fake = D(fake).reshape(-1) # flatten
            gp = gradient_penalty(D, real, fake, batch_size, num_channels, sequence_length)
            critic_loss = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            D.zero_grad()
            critic_loss.backward(retain_graph=True)
            critic_optimizer.step()

        generated_data = fake
        gen_fake = D(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        G.zero_grad()
        loss_gen.backward()
        gen_optimizer.step()

        critic_err.append(critic_loss.item())
        gen_err.append(loss_gen.item())

        if epoch % print_interval == 0:
            print("Epoch %s: Critic error (%s) Generator err (%s)" % (epoch, critic_err, gen_err))

    return critic_err, gen_err, generated_data, D, G

def weights_init(model):
    for m in model.modules():
      if isinstance(m, (nn.Conv1d, nn.ConvTranspose2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

def sample_normal_distribution(i, j, k):
    mu = 0
    sigma = 1
    return torch.Tensor(np.random.normal(mu, sigma, (i, j, k)))

def gradient_penalty(D, real, fake, batch_size, channel, sequence_length):
    alpha = torch.rand((batch_size, 1, 1)).repeat(1, channel, sequence_length)
    interpolated_seq = real * alpha + fake * (1 - alpha)

    # Calculate discriminator scores
    mixed_scores = D(interpolated_seq)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_seq,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty
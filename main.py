from matplotlib import pyplot as plt
import torch
import numpy as np

import train as train_model

def start_experiment():

    # trainig data should be in shape (batch_size, num_channels, sequence_length), ideally a numpy
    training_data = None

    # change these parameters
    batch_size = 60
    num_channels = 64
    sequence_length = 3000 
    num_epochs = 500

    critic_err, gen_err, generated_data, disc, gen = train_model(training_data, batch_size, num_channels, sequence_length, num_epochs)
    
    torch.save(gen, 'gen.pt')
    torch.save(disc, 'disc.pt')

    data = generated_data.detach().numpy()
    np.save('generated_data.npy', data)

    # plot the generated data
    num_rows = 6
    num_cols = 10
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 12))
    for i, ax in enumerate(axes.flatten()):
        y = generated_data[i, :, :]
        ax.plot(y.tolist()[0])
    plt.tight_layout()
    plt.show()

    # plotting G vs D losses
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(gen_err,label="Generator")
    plt.plot(critic_err,label="Critic")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, nz, num_users, user_embedding_dim=10):
        super(Generator, self).__init__()

        # Embedding layer to transform userID into a vector representation
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)

        # The input size is now the latent vector (nz) + the embedding of the userID
        self.fc1 = nn.Linear(nz + user_embedding_dim, 512 * 4 * 4)  # Ensure correct input size here

        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, user_id):
        # userID = torch.tensor(userID, dtype=torch.long)
        # Embed the userID
        user_embedded = self.user_embedding(user_id)

        # Ensure user_embedded has the correct shape to concatenate with z
        user_embedded = user_embedded.unsqueeze(2).unsqueeze(3)  # Shape: (batch_size, embedding_dim, 1, 1)

        # Concatenate the latent vector z with the userID embedding
        input_vector = torch.cat((z, user_embedded), dim=1)

        # Pass through fully connected layer to reshape the input for the convolutional layers
        x = self.fc1(input_vector.view(input_vector.size(0), -1))  # Flatten the input
        x = x.view(x.size(0), 512, 4, 4)  # Reshape to (batch_size, 512, 4, 4) before passing to transposed convs

        # Pass through the transposed convolutions to generate the image
        return self.main(x)

import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):

    def __init__(self, hparams, input_size=28 * 28, latent_dim=20):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size
        self.hparams = hparams

        hidden_dims = self.hparams.get("encoder_hidden_dims", [512, 256])
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], self.latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=20, output_size=28 * 28):
        super().__init__()

        self.hparams = hparams
        self.latent_dim = latent_dim
        self.output_size = output_size

        hidden_dims = self.hparams.get("decoder_hidden_dims", [256, 512])
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], self.output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self, hparams, encoder, decoder):
        super().__init__()
        self.hparams = hparams
        self.encoder = encoder
        self.decoder = decoder
        self.device = hparams.get(
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.set_optimizer()

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def set_optimizer(self):
        learning_rate = self.hparams.get("learning_rate", 1e-3)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def training_step(self, batch, loss_func):
        self.train()
        self.to(self.device)

        X = batch.to(self.device)
        flattened_X = X.view(X.shape[0], -1)

        self.optimizer.zero_grad()
        reconstruction = self.forward(flattened_X)
        loss = loss_func(reconstruction, flattened_X)
        loss.backward()
        self.optimizer.step()
        return loss

    def validation_step(self, batch, loss_func):
        self.eval()
        self.to(self.device)

        with torch.no_grad():
            X = batch.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            loss = loss_func(reconstruction, flattened_X)
        return loss

    def getReconstructions(self, loader=None):
        assert loader is not None, "Please provide a dataloader for reconstruction"

        self.eval()
        self.to(self.device)

        reconstructions = []
        with torch.no_grad():
            for batch in loader:
                X = batch.to(self.device)
                flattened_X = X.view(X.shape[0], -1)
                reconstruction = self.forward(flattened_X)
                reconstructions.append(
                    reconstruction.view(-1, 28, 28).cpu().numpy()
                )

        return np.concatenate(reconstructions, axis=0)


class Classifier(nn.Module):

    def __init__(self, hparams, encoder):
        super().__init__()
        self.hparams = hparams
        self.encoder = encoder
        self.device = hparams.get(
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        hidden_dim = self.hparams.get("classifier_hidden_dim", 128)
        num_classes = self.hparams.get("num_classes", 10)

        self.model = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.set_optimizer()

    def forward(self, x):
        features = self.encoder(x)
        return self.model(features)

    def set_optimizer(self):
        learning_rate = self.hparams.get("learning_rate", 1e-3)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def getAcc(self, loader=None):
        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self.to(self.device)

        scores = []
        labels = []

        with torch.no_grad():
            for batch in loader:
                X, y = batch
                X = X.to(self.device)
                flattened_X = X.view(X.shape[0], -1)
                score = self.forward(flattened_X)
                scores.append(score.detach().cpu().numpy())
                labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc


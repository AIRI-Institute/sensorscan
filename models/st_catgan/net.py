import torch
import torch.nn as nn
from torch.optim import Adam
from pytorch_lightning import LightningModule
from torchvision.utils import make_grid
from models.st_catgan.loss import gen_loss, disc_loss
from sklearn.metrics.cluster import normalized_mutual_info_score

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    def forward(self, x):
        return x.reshape(-1, *self.shape)

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)
    
class CatGAN(LightningModule):
    def __init__(
        self, 
        in_dim, 
        h_dim,
        latent_dim,
        n_sensors,
        n_types,
        batch_size,
        lr,
        n_disc,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.gen = nn.Sequential(
            Reshape(in_dim, 3, 3),

            nn.ConvTranspose2d(in_channels=in_dim, out_channels=h_dim, kernel_size=(2, 2), stride=2), 
            nn.BatchNorm2d(num_features=h_dim),
            nn.ReLU(), 

            nn.ConvTranspose2d(in_channels=h_dim, out_channels=h_dim, kernel_size=(2, 2), stride=2), 
            nn.BatchNorm2d(num_features=h_dim), 
            nn.ReLU(), 

            nn.ConvTranspose2d(in_channels=h_dim, out_channels=h_dim, kernel_size=(3, 3), stride=1), 
            nn.BatchNorm2d(num_features=h_dim),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=h_dim, out_channels=n_sensors, kernel_size=(3, 3), stride=1), 
            nn.Tanh(),
        )
        self.disc = disc = nn.Sequential(
            nn.Conv2d(in_channels=n_sensors, out_channels=h_dim, kernel_size=(3, 3), stride=1), 
            nn.LeakyReLU(), 

            nn.Conv2d(in_channels=h_dim, out_channels=h_dim, kernel_size=(2, 2), stride=2), 
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=h_dim, out_channels=in_dim, kernel_size=(2, 2), stride=2), 
            nn.LeakyReLU(),

            nn.Flatten(),
            nn.Linear(latent_dim, n_types),
            nn.Softmax(dim=1),
        )

    def training_step(self, batch, batch_idx):

        gen_opt, disc_opt = self.optimizers()

        Zxx, _, _ = batch
        batch_size = Zxx.shape[0]
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        
        for _ in range(self.hparams.n_disc):
            real_pred = self.disc(Zxx)
            with torch.no_grad():
                fake_Zxx = self.gen(z)
            fake_pred = self.disc(fake_Zxx)
            d_loss = disc_loss(real_pred, fake_pred)

            disc_opt.zero_grad()
            self.manual_backward(d_loss)
            disc_opt.step()

        fake_Zxx = self.gen(z)
        fake_pred = self.disc(fake_Zxx)
        g_loss = gen_loss(fake_pred)
        
        gen_opt.zero_grad()
        self.manual_backward(g_loss)
        gen_opt.step()

        self.log_dict({'gen_loss': g_loss, 'disc_loss': d_loss})
        
    def validation_step(self, batch, batch_idx):
        Zxx, _, label = batch
        batch_size = Zxx.shape[0]
        with torch.no_grad():
            pred = self.disc(Zxx).cpu().numpy().argmax(1)
        score = normalized_mutual_info_score(label, pred)
        self.log('nmi_score', score, prog_bar=True, batch_size=batch_size)
        self.log('hp_metric', score, batch_size=batch_size)
        
    def on_validation_epoch_end(self):
        z = torch.randn(1, self.hparams.latent_dim, device=self.device)
        with torch.no_grad():
            fake_Zxx = self.gen(z).cpu()
        grid = make_grid(fake_Zxx.transpose(0, 1), nrow=8)
        self.logger.experiment.add_image('fake_spectrograms', grid, self.current_epoch)

    def configure_optimizers(self):
        gen_opt = Adam(self.gen.parameters(), lr=self.hparams.lr)
        disc_opt = Adam(self.disc.parameters(), lr=self.hparams.lr)
        return gen_opt, disc_opt


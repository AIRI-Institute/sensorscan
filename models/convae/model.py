import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
        
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
        
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)
    

class ConvAE(nn.Module):
    
    def __init__(self):
        super(ConvAE, self).__init__()

        self.encoder = nn.Sequential(
                                     nn.Upsample(scale_factor = (1, 2)),
                                     nn.Conv2d(1, 32, 3, stride = 2, padding = 1),
                                     nn.LeakyReLU(0.2),
                                     nn.Conv2d(32, 64, 3, 2, 1),
                                     nn.LeakyReLU(0.2),
                                     nn.Flatten(),
                                     nn.Linear(24000, 25),
                                     nn.ReLU(),
                                    )
        self.decoder = nn.Sequential(
                                     nn.Linear(25, 24000),
                                     Reshape(64, 25, 15),
                                     nn.ConvTranspose2d(64, 64, 3, 2, 1, output_padding = 1),
                                     nn.LeakyReLU(0.2),
                                     nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding = 1),
                                     nn.LeakyReLU(0.2),
                                     nn.Conv2d(32, 1, 3, 1 , 1),
                                     nn.MaxPool2d((1, 2))
                                    )
        
    def forward(self, x):
        features = self.encoder(x)
        x =  self.decoder(features)

        return features, x
    
class CNN(nn.Module):

    def __init__(self, encoder, n_types):
        super(CNN, self).__init__()

        self.encoder = encoder

        self.class_head = nn.Sequential(nn.Linear(25, 64),
                           nn.BatchNorm1d(64),
                           nn.ReLU(),
                           nn.Linear(64, n_types))
        
    def forward(self, x):
        embedings = self.encoder(x)
        logits = self.class_head(embedings)

        return logits
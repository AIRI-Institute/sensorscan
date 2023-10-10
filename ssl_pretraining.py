import os

import cuml
import hydra
import matplotlib.pyplot as plt
import torch
from data_utils import build_pretraining_dataloader
from eval_utils import pretraining_visual_evaluation
from model import build_encoder
from omegaconf import OmegaConf
from optim import build_pretraining_optim
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from train_utils import train_ssl_epoch


@hydra.main(version_base=None, config_path="pretraining")
def ssl_pretraining(cfg):

    print(OmegaConf.to_yaml(cfg))
    plt.rcParams["figure.figsize"] = (10,10)

    train_loader, visualization_loader = build_pretraining_dataloader(cfg)

    model = build_encoder(cfg)
    loss_fn, optimizer = build_pretraining_optim(cfg, model)

    experiment_folder = 'experiments/'
    experiment_path = os.path.join(experiment_folder, cfg.experiment_name, cfg.run_dir)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path, exist_ok= True)

    OmegaConf.save(cfg, os.path.join(experiment_path, "config.yaml"))

    tsne = Pipeline([('scaler', StandardScaler()), ('tsne',cuml.TSNE(n_components=2, verbose = 5, n_iter = 8000, perplexity= 40, learning_rate = 300, method= 'barnes_hut'))])

    for epoch in range(cfg.epochs):
    
        avg_loss = train_ssl_epoch(cfg, model, train_loader, loss_fn, optimizer)
        print(f'Epoch {epoch}: loss = {avg_loss:10.8f}')

        scatter = pretraining_visual_evaluation(cfg, model, visualization_loader, tsne)
        fig = scatter.get_figure()
        fig.savefig(os.path.join(experiment_path, f"{epoch}_visualization.png"))
        plt.close(fig)

        torch.save(model.state_dict(), os.path.join(experiment_path, f"{epoch}_weights.pth"))

if __name__ == '__main__':
    ssl_pretraining()

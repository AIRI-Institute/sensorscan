import os

import cuml
import hydra
import matplotlib.pyplot as plt
import torch
from data_utils import build_neighbour_loader
from eval_utils import scan_visual_evaluation
from model import build_encoder, build_clustering
from omegaconf import OmegaConf
from optim import build_scan_optim
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from train_utils import train_scan_epoch

@hydra.main(version_base=None, config_path=".")
def scan_training(cfg):

    print(OmegaConf.to_yaml(cfg))
    plt.rcParams["figure.figsize"] = (10,10)

    encoder = build_encoder(cfg.pretraining)
    encoder.load_state_dict(torch.load(cfg.pretrained_path, map_location= cfg.device))

    neighbor_loader, visualization_loader = build_neighbour_loader(cfg, encoder)

    clustering_model = build_clustering(cfg)

    loss_fn, encoder_optimizer, clustering_optimizer = build_scan_optim(cfg, encoder, clustering_model)

    experiment_folder = 'experiments/'
    experiment_path = os.path.join(experiment_folder, cfg.experiment_name, cfg.run_dir)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path, exist_ok= True)

    OmegaConf.save(cfg, os.path.join(experiment_path, "config.yaml"))

    tsne = Pipeline([('scaler', StandardScaler()), ('tsne',cuml.TSNE(n_components=2, verbose = 5, n_iter = 8000, perplexity= 40, learning_rate = 300, method= 'barnes_hut'))])

    for epoch in range(cfg.epochs):
    
        avg_loss = train_scan_epoch(cfg, epoch, encoder, clustering_model, neighbor_loader, loss_fn, encoder_optimizer, clustering_optimizer)
        print(f'Epoch {epoch}: loss = {avg_loss:10.8f}')

        scatter = scan_visual_evaluation(cfg, encoder, clustering_model, visualization_loader, tsne)
        fig = scatter.get_figure()
        fig.savefig(os.path.join(experiment_path, f"{epoch}_visualization.png"))
        plt.close(fig)

        torch.save(encoder.state_dict(), os.path.join(experiment_path, f"{epoch}_encoder_weights.pth"))
        torch.save(clustering_model.state_dict(), os.path.join(experiment_path, f"{epoch}_clustering_weights.pth"))

if __name__ == '__main__':
    scan_training()

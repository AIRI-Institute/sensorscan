import hydra
import torch
from data_utils import build_test_loader
from model import build_encoder, build_clustering
from omegaconf import OmegaConf
from eval_utils import metric_evaluation

@hydra.main(version_base=None, config_path=".")
def evaluate(cfg):

    print(OmegaConf.to_yaml(cfg))

    encoder = build_encoder(cfg.pretraining)
    encoder.load_state_dict(torch.load(cfg.encoder_path, map_location= cfg.device))
    clustering_model = build_clustering(cfg)
    clustering_model.load_state_dict(torch.load(cfg.clustering_path, map_location= cfg.device))

    test_loader = build_test_loader(cfg)

    metric_evaluation(cfg, encoder, clustering_model, test_loader)

if __name__ == '__main__':
    evaluate()
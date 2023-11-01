import torch
import numpy as np
from models.sensorscan.data_utils import build_pretraining_dataloader, build_neighbour_loader
from models.sensorscan.model import build_encoder, build_clustering, SensorSCAN
from models.sensorscan.optim import build_pretraining_optim, build_scan_optim
from models.sensorscan.train_utils import train_ssl_epoch, train_scan_epoch
from tqdm.auto import tqdm
import pandas as pd
from fddbenchmark import FDDDataset, FDDDataloader
import logging
from utils import exclude_columns, make_rieth_imbalance, normalize

def run(cfg):
    
    dataset_name = cfg.dataset
    window_size = cfg.window_size
    step_size = cfg.step_size
    random_seed = cfg.random_seed
    eval_batch_size = cfg.eval_batch_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.device = device
    cfg.pretraining.device = device

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    logging.info('Creating dataset')

    dataset = FDDDataset(name=dataset_name)
    dataset.df = exclude_columns(dataset.df)
    normalize(dataset)
    if dataset_name == 'rieth_tep':
        dataset.train_mask = make_rieth_imbalance(dataset.train_mask)

    logging.info('Creating dataloaders')

    train_loader = FDDDataloader(
        dataframe=dataset.df, 
        mask=dataset.train_mask, 
        label=dataset.label,
        window_size=window_size, 
        step_size=step_size,
        use_minibatches=True, 
        batch_size=eval_batch_size,
        shuffle=True,
    )

    test_loader = FDDDataloader(
        dataframe=dataset.df, 
        mask=dataset.test_mask, 
        label=dataset.label,
        window_size=window_size, 
        step_size=step_size,
        use_minibatches=True, 
        batch_size=eval_batch_size,
        shuffle=False,
    )

    if cfg.path_to_model is None:
        logging.info('Pretraining encoder')
        # pretraining
        pretraining_loader = build_pretraining_dataloader(cfg.pretraining)
        encoder = build_encoder(cfg.pretraining)
        loss_fn, optimizer = build_pretraining_optim(cfg.pretraining, encoder)
        for epoch in range(cfg.pretraining.epochs):
            avg_loss = train_ssl_epoch(cfg.pretraining, encoder, pretraining_loader, loss_fn, optimizer)
            print(f'Epoch {epoch}: loss = {avg_loss:10.8f}')

        logging.info('Training SCAN')
        # scan training
        neighbor_loader = build_neighbour_loader(cfg, encoder)
        clustering_model = build_clustering(cfg)
        loss_fn, encoder_optimizer, clustering_optimizer = build_scan_optim(cfg, encoder, clustering_model)
        for epoch in range(cfg.epochs):
            avg_loss = train_scan_epoch(cfg, epoch, encoder, clustering_model, neighbor_loader, loss_fn, encoder_optimizer, clustering_optimizer)
            print(f'Epoch {epoch}: loss = {avg_loss:10.8f}')

        sensorscan = SensorSCAN(encoder, clustering_model, device=cfg.device)
        cfg.path_to_model = f'saved_models/sensorscan_{dataset_name}.pth'
        torch.save(sensorscan.state_dict(), cfg.path_to_model)
    
    sensorscan = SensorSCAN(build_encoder(cfg.pretraining), build_clustering(cfg), device=cfg.device)
    sensorscan.load_state_dict(torch.load(cfg.path_to_model, map_location=cfg.device))

    logging.info('Getting predictions on train')
    sensorscan.eval()
    train_pred = []
    train_label = []
    for X, time_index, label in tqdm(train_loader, desc='Getting predictions on train'):
        with torch.no_grad():
            X = torch.FloatTensor(X).to(cfg.device)
            pred = sensorscan(X).cpu().numpy().argmax(1)
        train_pred.append(pd.Series(pred, index=time_index))
        train_label.append(pd.Series(label, index=time_index))
    train_pred = pd.concat(train_pred)
    train_label = pd.concat(train_label).astype('int')

    logging.info('Getting predictions on test')
    test_pred = []
    test_label = []
    for X, time_index, label in tqdm(test_loader, desc='Getting predictions on test'):
        with torch.no_grad():
            X = torch.FloatTensor(X).to(cfg.device)
            pred = sensorscan(X).cpu().numpy().argmax(1)
        test_pred.append(pd.Series(pred, index=time_index))
        test_label.append(pd.Series(label, index=time_index))
    test_label = pd.concat(test_label).astype('int')
    test_pred = pd.concat(test_pred)

    return train_pred, train_label, test_pred, test_label

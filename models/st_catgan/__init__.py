import torch
from pytorch_lightning import Trainer, seed_everything
from fddbenchmark import FDDDataset
from utils import make_rieth_imbalance
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pandas as pd
from models.st_catgan.net import CatGAN
from models.st_catgan.data import STFTDataloader
import logging

def run(args):
    
    dataset_name = args.dataset
    window_size = args.window_size
    step_size = args.step_size
    random_seed = args.random_seed
    eval_batch_size = args.eval_batch_size

    in_dim = args.in_dim
    h_dim = args.h_dim
    batch_size = args.batch_size
    train_step_size = args.train_step_size
    n_epochs = args.n_epochs
    lr = args.lr
    nperseg = args.nperseg
    noverlap = args.noverlap
    path_to_model = args.path_to_model
    log_every_n_steps = args.log_every_n_steps
    n_disc = args.n_disc
    
    latent_dim = in_dim*3*3
    if dataset_name == 'small_tep':
        n_types = 21
        n_sensors = 52
    elif dataset_name == 'rieth_tep':
        n_types = 21
        n_sensors = 52
    elif dataset_name == 'reinartz_tep':
        n_types = 29
        n_sensors = 52

    if random_seed is not None:
        seed_everything(random_seed)
    
    logging.info('Creating dataset')

    dataset = FDDDataset(name=dataset_name)
    scaler = MinMaxScaler((-1, 1))
    scaler.fit(dataset.df[dataset.train_mask])
    dataset.df.iloc[:, :] = scaler.transform(dataset.df)
    if dataset_name == 'rieth_tep':
        dataset.train_mask = make_rieth_imbalance(dataset.train_mask)

    logging.info('Creating dataloaders')

    in_cache = True
    if path_to_model is not None:
        in_cache = False

    train_loader = STFTDataloader(
        nperseg=nperseg, 
        noverlap=noverlap,
        in_cache=in_cache,
        dataframe=dataset.df, 
        mask=dataset.train_mask, 
        label=dataset.label,
        window_size=window_size, 
        step_size=train_step_size,
        use_minibatches=True,
        batch_size=batch_size,
        shuffle=True, 
        random_state=random_seed,
    )
    
    test_loader = STFTDataloader(
        nperseg=nperseg, 
        noverlap=noverlap,
        in_cache=False,
        dataframe=dataset.df,
        mask=dataset.test_mask,
        label=dataset.label,
        window_size=window_size, 
        step_size=step_size, 
        use_minibatches=True,
        batch_size=eval_batch_size,
    )

    if path_to_model is not None:
        logging.info('Loading model')
        catgan = CatGAN.load_from_checkpoint(path_to_model)
    else:
        logging.info('Training model')
        catgan = CatGAN(
            in_dim, h_dim, latent_dim, n_sensors, n_types, batch_size, lr, n_disc)
        trainer = Trainer(
            accelerator='auto',
            max_epochs=n_epochs,
            log_every_n_steps=log_every_n_steps,
        )
        trainer.fit(
            model=catgan, 
            train_dataloaders=train_loader,
            val_dataloaders=train_loader,
        )

    logging.info('Getting predictions on train')
    catgan.eval()
    train_pred = []
    train_label = []
    for Zxx, time_index, label in tqdm(train_loader, desc='Getting predictions on train'):
        with torch.no_grad():
            pred = catgan.disc(Zxx).numpy().argmax(1)
        train_pred.append(pd.Series(pred, index=time_index))
        train_label.append(pd.Series(label, index=time_index))
    train_pred = pd.concat(train_pred)
    train_label = pd.concat(train_label).astype('int')

    logging.info('Getting predictions on test')
    test_pred = []
    test_label = []
    for Zxx, time_index, label in tqdm(test_loader, desc='Getting predictions on test'):
        with torch.no_grad():
            pred = catgan.disc(Zxx).numpy().argmax(1)
        test_pred.append(pd.Series(pred, index=time_index))
        test_label.append(pd.Series(label, index=time_index))
    test_label = pd.concat(test_label).astype('int')
    test_pred = pd.concat(test_pred)
    
    return train_pred, train_label, test_pred, test_label

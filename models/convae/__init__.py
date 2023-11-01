import torch
import numpy as np
from fddbenchmark import FDDDataset, FDDDataloader
from utils import exclude_columns, make_rieth_imbalance, normalize
from tqdm import tqdm
import pandas as pd
from models.convae.model import ConvAE, CNN, init_weights
from models.convae.utils import pretraining, build_pseudolabels, finetune
import logging

def run(args):

    dataset_name = args.dataset
    window_size = args.window_size
    step_size = args.step_size
    random_seed = args.random_seed
    eval_batch_size = args.eval_batch_size

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    if dataset_name == 'small_tep':
        n_types = 21
    elif dataset_name == 'rieth_tep':
        n_types = 21
    elif dataset_name == 'reinartz_tep':
        n_types = 29

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretraining_batch_size = args.pretraining_batch_size
    finetuning_batch_size = args.finetuning_batch_size
    pretraining_lr = args.pretraining_lr
    finetuning_lr = args.finetuning_lr
    pretraining_epochs = args.pretraining_n_epochs
    finetuning_epochs = args.finetuning_n_epochs
    labeling_step_size = args.labeling_step_size
    path_to_model = args.path_to_model

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
        batch_size=pretraining_batch_size,
        shuffle=True
    )

    labeling_loader = FDDDataloader(
        dataframe=dataset.df, 
        mask=dataset.train_mask, 
        label=dataset.label,
        window_size=window_size, 
        step_size=labeling_step_size,
        use_minibatches=True, 
        batch_size=pretraining_batch_size,
        shuffle=False
    )

    test_loader = FDDDataloader(
        dataframe=dataset.df, 
        mask=dataset.test_mask, 
        label=dataset.label,
        window_size=window_size, 
        step_size=step_size,
        use_minibatches=True, 
        batch_size=eval_batch_size,
        shuffle=False
    )

    if path_to_model is not None:
        cnn_model = CNN(ConvAE().encoder, n_types)
        cnn_model = cnn_model.to(device)
        cnn_model.load_state_dict(torch.load(path_to_model, map_location=device))
    else:
        logging.info('Training model')
        conv_ae = ConvAE()
        conv_ae.apply(init_weights)
        conv_ae = conv_ae.to(device)
        
        pretraining(conv_ae, train_loader, pretraining_epochs, pretraining_lr, device)

        time_series_train, pseudolabels = build_pseudolabels(conv_ae, labeling_loader, n_types, device)

        finetuning_dataset = torch.utils.data.TensorDataset(time_series_train, pseudolabels)
        finetuning_loader = torch.utils.data.DataLoader(finetuning_dataset, 
                                                        batch_size = finetuning_batch_size, 
                                                        shuffle = True, 
                                                        num_workers = 4)

        cnn_model = CNN(conv_ae.encoder, n_types)
        cnn_model.class_head.apply(init_weights)
        cnn_model = cnn_model.to(device)

        finetune(cnn_model, finetuning_loader, finetuning_epochs, finetuning_lr, device)

        torch.save(cnn_model.state_dict(), f'saved_models/convae_{dataset_name}.pth')

    logging.info('Getting predictions on train')
    cnn_model.eval()
    train_pred = []
    train_label = []
    for X, time_index, label in tqdm(train_loader, desc='Getting predictions on train'):
        with torch.no_grad():
            X = torch.FloatTensor(X)[:, None].to(device)
            pred = cnn_model(X).cpu().numpy().argmax(1)
        train_pred.append(pd.Series(pred, index=time_index))
        train_label.append(pd.Series(label, index=time_index))
    train_pred = pd.concat(train_pred)
    train_label = pd.concat(train_label).astype('int')

    logging.info('Getting predictions on test')
    test_pred = []
    test_label = []
    for X, time_index, label in tqdm(test_loader, desc='Getting predictions on test'):
        with torch.no_grad():
            X = torch.FloatTensor(X)[:, None].to(device)
            pred = cnn_model(X).cpu().numpy().argmax(1)
        test_pred.append(pd.Series(pred, index=time_index))
        test_label.append(pd.Series(label, index=time_index))
    test_label = pd.concat(test_label).astype('int')
    test_pred = pd.concat(test_pred)
    
    return train_pred, train_label, test_pred, test_label

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from fddbenchmark import FDDDataset, FDDDataloader
from utils import exclude_columns, make_rieth_imbalance
from joblib import dump, load
import logging

def run(args):

    dataset_name = args.dataset
    window_size = args.window_size
    step_size = args.step_size
    random_seed = args.random_seed
    eval_batch_size = args.eval_batch_size
    
    latent_dim = args.latent_dim
    train_step_size = args.train_step_size
    if dataset_name == 'small_tep':
        n_types = 21
    elif dataset_name == 'rieth_tep':
        n_types = 21
    elif dataset_name == 'reinartz_tep':
        n_types = 29
    path_to_model = args.path_to_model
    
    logging.info('Creating dataset')

    dataset = FDDDataset(name=dataset_name)
    dataset.df = exclude_columns(dataset.df)
    if dataset_name == 'rieth_tep':
        dataset.train_mask = make_rieth_imbalance(dataset.train_mask)

    logging.info('Creating dataloaders')

    train_loader = FDDDataloader(
        dataframe=dataset.df, 
        mask=dataset.train_mask, 
        label=dataset.label,
        window_size=window_size, 
        step_size=train_step_size,
    )
    
    test_loader = FDDDataloader(
        dataframe=dataset.df, 
        mask=dataset.test_mask, 
        label=dataset.label, 
        window_size=window_size, 
        step_size=step_size,
        use_minibatches=True,
        batch_size=eval_batch_size
    )

    for train_ts, train_index, train_label in train_loader:
        break
    train_ts = train_ts.reshape(train_ts.shape[0], -1)

    if path_to_model is not None:
        logging.info('Loading model')
        model = load(path_to_model)
    else:
        logging.info('Training model')
        model = make_pipeline(
            StandardScaler(), 
            PCA(n_components=latent_dim, random_state=random_seed), 
            KMeans(n_clusters=n_types, verbose=1, random_state=random_seed)
        )
        model.fit(train_ts)
        logging.info('Saving model')
        dump(model, f'saved_models/pca_kmeans_{dataset_name}.joblib')

    logging.info('Getting predictions on train')
    train_pred = pd.Series(
        model.predict(train_ts), 
        index=train_index
    )

    logging.info('Getting predictions on test')
    _pred = []
    _label = []
    for test_ts, test_index, test_label in tqdm(test_loader):
        test_ts = test_ts.reshape(test_ts.shape[0], -1)
        test_pred = model.predict(test_ts)
        _pred.append(pd.Series(test_pred, index=test_index))
        _label.append(pd.Series(test_label, index=test_index))
    test_label = pd.concat(_label).astype('int')
    test_pred = pd.concat(_pred)

    return train_pred, train_label, test_pred, test_label

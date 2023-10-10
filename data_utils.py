import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from fddbenchmark import FDDDataloader, FDDDataset
from tqdm import tqdm


def build_pretraining_dataloader(cfg, visualize = True):

    dataset = _build_fdd_dataset(cfg)

    train_dataset = PretraingDataset(cfg, dataset)

    train_dataloader = torch.utils.data.DataLoader(dataset= train_dataset,
                              batch_size= cfg.train_batch_size,
                              shuffle=True,
                              num_workers= 4,
                              pin_memory=True, drop_last = True)

    if not visualize:
        return train_dataloader
    else: 
        torch_dataset = TrainingDataset(cfg, dataset)
        subsample_idxs = np.random.choice(np.arange(0, len(torch_dataset)), min(len(torch_dataset), cfg.visualize_size), replace = False)
        torch_dataset_subset = torch.utils.data.Subset(torch_dataset, subsample_idxs)

        visualization_dataloader = torch.utils.data.DataLoader(dataset= torch_dataset_subset,
                              batch_size= cfg.visualize_batch_size,
                              shuffle=False,
                              num_workers= 4,
                              pin_memory=True, drop_last = False)

        return train_dataloader, visualization_dataloader

def build_neighbour_loader(cfg, encoder, visualize = True):
    dataset = _build_fdd_dataset(cfg.pretraining)
    
    train_dataset = TrainingDataset(cfg.pretraining, dataset)
    train_subsample_idxs = np.random.choice(np.arange(0, len(train_dataset)), min(len(train_dataset), cfg.neighbour_dataset_size), replace = False)
    train_dataset_subset = torch.utils.data.Subset(train_dataset, train_subsample_idxs)

    neighbor_dataset = NeighborDataset.build_with_encoder(cfg, train_dataset_subset, encoder)

    neighbor_loader = torch.utils.data.DataLoader(dataset= neighbor_dataset,
                              batch_size= cfg.scan_batch_size,
                              shuffle=True,
                              num_workers= 4,
                              pin_memory=True, drop_last = True)
    
    if not visualize:
        return neighbor_loader
    else:
        subsample_idxs = np.random.choice(np.arange(0, len(train_dataset)), min(len(train_dataset), cfg.pretraining.visualize_size), replace = False)
        torch_dataset_subset = torch.utils.data.Subset(train_dataset, subsample_idxs)

        visualization_dataloader = torch.utils.data.DataLoader(dataset= torch_dataset_subset,
                              batch_size= cfg.pretraining.visualize_batch_size,
                              shuffle=False,
                              num_workers= 4,
                              pin_memory=True, drop_last = False)

        return neighbor_loader, visualization_dataloader
    
def build_test_loader(cfg):
    dataset = _build_fdd_dataset(cfg.pretraining)

    test_dataset = TrainingDataset(cfg.pretraining, dataset, train= False)
    return torch.utils.data.DataLoader(dataset= test_dataset,
                              batch_size= cfg.scan_batch_size,
                              shuffle=False,
                              num_workers= 4,
                              pin_memory=True, drop_last = False)

class PretraingDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, fdd_dataset):
        self.fdd_dataset = fdd_dataset

        self.fddloader = FDDDataloader(
            dataframe= self.fdd_dataset.df,
            labels= self.fdd_dataset.labels,
            mask= self.fdd_dataset.train_mask,
            window_size=cfg.dataset_window_size,
            step_size=cfg.step_size,
            minibatch_training=True,
            batch_size= 1,
            shuffle= False
        )

        self.augmentor = Augmentor(cfg.augmentor)
        
    def __len__(self):
        return len(self.fddloader)
    
    def __getitem__(self, idx):
        window, _, label = self.fddloader[idx]
        X_weak, targets_weak, target_masks_weak, X_strong, targets_strong, target_masks_strong = self.augmentor(window[0])
        
        label = torch.LongTensor([label.iloc[0].astype('int')])

        return X_weak, targets_weak, target_masks_weak, X_strong, targets_strong, target_masks_strong, label
    
class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, fdd_dataset, train = True):
        self.fdd_dataset = fdd_dataset
        self.train = train
        
        if train:
            self.fddloader = FDDDataloader(
                dataframe= self.fdd_dataset.df,
                labels= self.fdd_dataset.labels,
                mask= self.fdd_dataset.train_mask,
                window_size=cfg.model_input_length,
                step_size=cfg.step_size,
                minibatch_training=True,
                batch_size= 1,
                shuffle= False
            )
        else:
            self.fddloader = FDDDataloader(
                dataframe= self.fdd_dataset.df,
                labels= self.fdd_dataset.labels,
                mask= self.fdd_dataset.test_mask,
                window_size= cfg.model_input_length,
                step_size= cfg.step_size,
                minibatch_training=True,
                batch_size= 1,
                shuffle= False
            )

    def __len__(self):
        return len(self.fddloader)

    def __getitem__(self, idx):
        window, _, label = self.fddloader[idx]

        window = torch.FloatTensor(window[0])
        label = torch.LongTensor([label.iloc[0].astype('int')])

        return window, label
    
class NeighborDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, neighbor_indices):
        super(NeighborDataset, self).__init__()

        self.dataset = dataset
        self.neighbor_indices = neighbor_indices

    @classmethod
    def build_with_encoder(cls, cfg, dataset, encoder):
        encoder = encoder.eval()

        embedings_list = []

        loader = torch.utils.data.DataLoader(dataset= dataset,
                              batch_size= cfg.scan_batch_size,
                              shuffle= False,
                              num_workers= 4,
                              pin_memory=True, drop_last = False)

        for X, _ in tqdm(loader, total= len(loader)):
            X = X.to(cfg.device)
            pad_mask = torch.ones(*X.shape[:-1], dtype = torch.bool, device = cfg.device)

            with torch.no_grad():
                _, embeddings = encoder(X, pad_mask)
            
            
            embedings_list.append(embeddings.cpu())

        embeddings = torch.cat(embedings_list, dim = 0).numpy()

        nbrs = NearestNeighbors(n_neighbors= cfg.num_neighbors, algorithm='ball_tree').fit(embeddings)
        _, idxs = nbrs.kneighbors(embeddings)
        idxs = idxs[:, 1:]

        return cls(dataset, idxs)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        X, label =  self.dataset[idx]
        
        neighbor_ind = np.random.choice(self.neighbor_indices[idx])
        neighbor, _ = self.dataset[neighbor_ind]
        
        return X, neighbor, label
    
    
def _build_fdd_dataset(cfg):

    dataset = FDDDataset(name= cfg.dataset_name)
    dataset.df.drop(['xmeas_23', 'xmeas_24', 'xmeas_25', 'xmeas_26',
           'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_30', 'xmeas_31', 'xmeas_32',
           'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36', 'xmeas_37', 'xmeas_38',
           'xmeas_39', 'xmeas_40', 'xmeas_41', 'xmv_5', 'xmv_9', 'xmeas_19'], axis = 1, inplace = True)
    
    mean = dataset.df[dataset.train_mask].mean(axis = 0)
    std = dataset.df[dataset.train_mask].std(axis = 0)
    dataset.df = (dataset.df - mean) / std

    return dataset
    
class Augmentor():

    def __init__(self, cfg):
        super(Augmentor, self).__init__()

        self.masking_ratio = cfg.masking_ratio
        self.mean_mask_length = cfg.mean_mask_length

        self.weak_jitter_sigma = cfg.weak_jitter_sigma
        self.weak_scaling_loc = cfg.weak_scaling_loc
        self.weak_scaling_sigma = cfg.weak_scaling_sigma
        self.strong_scaling_loc = cfg.strong_scaling_loc
        self.strong_scaling_sigma = cfg.strong_scaling_sigma
        self.strong_permute_max_segments = cfg.strong_permute_max_segments

    def __call__(self, X):
        
        X = X.transpose(1, 0) # (feat_dim, seq_length) array
        weak = jitter(scaling(X, loc = self.weak_scaling_loc, sigma= self.weak_scaling_sigma), sigma= self.weak_jitter_sigma)
        strong = scaling(permutation(X, max_segments= self.strong_permute_max_segments), loc = self.strong_scaling_loc, sigma= self.strong_scaling_sigma)
        
        weak = weak.transpose(1, 0) # (seq_length, feat_dim) array
        strong =  strong.transpose(1, 0) # (seq_length, feat_dim) array
        X = X.transpose(1, 0)
        
        mask_weak = noise_mask(weak, self.masking_ratio, self.mean_mask_length)  # (seq_length, feat_dim) boolean array
        mask_strong = noise_mask(strong, self.masking_ratio, self.mean_mask_length) # (seq_length, feat_dim) boolean array
        
        
        X_weak, target_masks_weak = torch.FloatTensor(weak), torch.BoolTensor(mask_weak)
        X_strong, target_masks_strong = torch.FloatTensor(strong), torch.BoolTensor(mask_strong)
        
        targets_weak = X_weak.clone()
        X_weak = X_weak * target_masks_weak  # mask input
        X_weak = compensate_masking(X_weak, target_masks_weak)
            
        targets_strong = X_strong.clone()
        X_strong = X_strong * target_masks_strong  
        X_strong = compensate_masking(X_strong, target_masks_strong)

        target_masks_weak = ~target_masks_weak  # inverse logic: 0 now means ignore, 1 means predict
        target_masks_strong = ~target_masks_strong
        
        return X_weak, targets_weak, target_masks_weak, X_strong, targets_strong, target_masks_strong

def noise_mask(X, masking_ratio, lm=3):

    mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])

    return mask

def geom_noise_mask_single(L, lm, masking_ratio):

    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask

def compensate_masking(X, mask):

    # number of unmasked elements of feature vector for each time step
    num_active = torch.sum(mask, dim=-1).unsqueeze(-1)  # (batch_size, seq_length, 1)
    # to avoid division by 0, set the minimum to 1
    num_active = torch.max(num_active, torch.ones(num_active.shape, dtype=torch.int16))  # (batch_size, seq_length, 1)
    return X.shape[-1] * X / num_active

def jitter(x, sigma): #0.08
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, loc, sigma): #0.1
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=loc, scale=sigma, size=(1, x.shape[1]))
 
    return x * factor

def permutation(x, max_segments):
    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(1, max_segments)

    if num_segs > 1:
        split_points = np.random.choice(x.shape[1] - 2, num_segs - 1, replace=False)
        split_points.sort()
        splits = np.split(orig_steps, split_points)

        warp = np.concatenate(np.random.permutation(splits)).ravel()
        return x[:, warp]
    else:
        return x
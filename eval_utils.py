import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from fddbenchmark import FDDEvaluator
from tqdm import tqdm

def pretraining_visual_evaluation(cfg, model, dataloader, dim_reduction):
    model = model.eval()

    predicts_train = []
    classes_train = []

    for X, labels in dataloader:
        X = X.to(cfg.device)
        pad_mask = torch.ones(*X.shape[:-1], dtype = torch.bool, device = cfg.device)

        with torch.no_grad():
            _, features = model(X, pad_mask)

        predicts_train.append(features.cpu())
        classes_train.append(labels)

    X_train = torch.cat(predicts_train, dim = 0).numpy()  
    Y_train = torch.cat(classes_train, dim = 0).numpy()

    X_train_tsne = dim_reduction.fit_transform(X_train)

    img = plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c = Y_train * 15, cmap = 'jet')

    return img

def scan_visual_evaluation(cfg, encoder, clustering_model, dataloader, dim_reduction):
    encoder = encoder.eval()
    clustering_model = clustering_model.eval()

    embedings_list = []
    cluster_list = []

    for X, _ in dataloader:
        X = X.to(cfg.device)
        pad_mask = torch.ones(*X.shape[:-1], dtype = torch.bool, device = cfg.device)

        with torch.no_grad():
            _, _, _, encoder_embedings, _ = encoder(X, pad_mask, return_all = True)
            final_embedings = clustering_model(encoder_embedings)

        embedings_list.append(encoder_embedings.cpu())
        _, pred_ind = torch.softmax(final_embedings, dim = 1).max(dim = 1)
        cluster_list.append(pred_ind.cpu())

    embedings = torch.cat(embedings_list, dim = 0).numpy()
    predicted_clusters = torch.cat(cluster_list, dim = 0).numpy()

    embedings_reduced = dim_reduction.fit_transform(embedings)

    img = plt.scatter(embedings_reduced[:, 0], embedings_reduced[:, 1], c = predicted_clusters * 15, cmap = 'jet')

    return img

def metric_evaluation(cfg, encoder, clustering_model, dataloader):
    encoder = encoder.eval()
    clustering_model = clustering_model.eval()

    evaluator = FDDEvaluator(cfg.pretraining.model_input_length)

    preds = []

    for X, _ in tqdm(dataloader, total= len(dataloader)):
        X = X.to(cfg.device)

        with torch.no_grad():
            pad_mask = torch.ones(*X.shape[:-1], dtype = torch.bool, device = cfg.device)
            _, _, _, encoder_embedings, _ = encoder(X, pad_mask, return_all = True)
            clutsering_output = clustering_model(encoder_embedings)

            _, pred = torch.softmax(clutsering_output, dim = 1).max(dim = 1)

        
        preds.append(pred.cpu())

    Y_pred = torch.cat(preds, dim = 0)

    true_labels = dataloader.dataset.fdd_dataset.labels[dataloader.dataset.fdd_dataset.test_mask]
    sample_values = true_labels.index.get_level_values('sample')
    true_labels = true_labels[(sample_values >= cfg.pretraining.model_input_length) & (sample_values < sample_values.max())]
    
    pred_labels = pd.Series(Y_pred, index= true_labels.index)
    pseudolabels = get_pseudolabels(true_labels, pred_labels)
    
    evaluator.print_metrics(true_labels.astype('int'), pd.Series(pseudolabels.astype('int'), index=true_labels.index))


def get_pseudolabels(Y_train, clustering_labels):
    pseudolabels = np.zeros_like(Y_train)
    for cluster_label in np.unique(clustering_labels):
        Y_cluster = Y_train[clustering_labels == cluster_label]
        unique_items, unique_counts = np.unique(Y_cluster, return_counts = True)
        
        find_result = np.where(unique_items == 0)[0]
        if find_result.size == 1 and (unique_counts[find_result[0]] / unique_counts.sum()) > (1 / (unique_items.size + 3)):
            pseudolabels[clustering_labels == cluster_label] = 0
        else:
            pseudo_Y = unique_items[unique_counts == unique_counts.max()]
            pseudolabels[clustering_labels == cluster_label] = np.random.choice(pseudo_Y)

    return pseudolabels
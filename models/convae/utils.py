import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def pretraining(model, dataloader, num_epochs, lr, device):
    l2_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    model.train()

    for epoch in range(num_epochs):
        loss_sum = torch.zeros(1, device= device)

        for x, _, _ in tqdm(dataloader, total=len(dataloader)):
            optimizer.zero_grad()

            x = torch.FloatTensor(x)[:, None]
            x = x.to(device)

            _, prediction = model(x)
            loss = l2_loss(prediction, x)
        
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.detach()

        print(f"Epoch {epoch}: loss {loss_sum.item() / len(dataloader)}")

def build_pseudolabels(model, dataloader, num_clusters, device):
    model.eval()

    features_list = []
    input_list = []

    for x, _, _ in tqdm(dataloader, total=len(dataloader)):
        x = torch.FloatTensor(x)[:, None]
        x = x.to(device)

        input_list.append(x.cpu())
        features, _ = model(x)
        features_list.append(features.detach().cpu())

    features_train = torch.cat(features_list, dim = 0).numpy()  
    time_series_train = torch.cat(input_list, dim = 0)


    tsne = Pipeline([('scaler', StandardScaler()), 
                        ('tsna',TSNE(n_components=2, 
                                     verbose = 5, 
                                     n_iter = 5000, 
                                     perplexity= 40, 
                                     learning_rate = 600, 
                                     method= 'barnes_hut'))])
    
    k_means = KMeans(num_clusters)

    features_train_tsne = tsne.fit_transform(features_train)
    
    features_train_tsne_scaled = tsne['scaler'].fit_transform(features_train_tsne)
    clustering_labels  = k_means.fit_predict(features_train_tsne_scaled)
    
    return time_series_train, torch.LongTensor(clustering_labels)


def finetune(model, loader, num_epochs, lr, device):
    model.train()

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(num_epochs):

        loss_sum = torch.zeros(1, device= device)
        
        true_labels = []
        pred_labels = []
        
        for n, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            
            x, y = x.to(device), y.to(device).squeeze()
            
            logits = model(x)
            
            loss = ce_loss(logits, y)
            loss.backward()
            
            optimizer.step()
            
            loss_sum += loss.detach()
            
            _, pred_ind = torch.softmax(logits.detach(), dim = 1).max(dim = 1)
            
            pred_labels.append(pred_ind.cpu())
            true_labels.append(y.cpu())
                
        pred_labels = torch.cat(pred_labels, dim = 0).numpy()
        true_labels = torch.cat(true_labels, dim = 0).numpy()
                
        print(f'Epoch {epoch}: loss = {loss_sum.item() / (n + 1):10.8f}')
        print(f"Acc : {accuracy_score(pred_labels, true_labels)}")

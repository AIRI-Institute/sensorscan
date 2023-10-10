import numpy as np
import torch


def build_pretraining_optim(cfg, model):
    l2_loss = MaskedMSELoss()
    ntx_loss = NTXentLoss(cfg.device, cfg.train_batch_size)

    def loss_fn(prediction, y, mask, embedings_weak, embedings_strong):
        return l2_loss(prediction, y, mask) + cfg.contrastive_weight *  ntx_loss(embedings_weak, embedings_strong)

    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.lr, weight_decay= cfg.weight_decay)

    return loss_fn, optimizer

def build_scan_optim(cfg, encoder, clustering_model):
    loss_fn = SCANLoss(cfg.entropy_weight)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr = cfg.encoder_lr)
    clustering_optimizer = torch.optim.Adam(clustering_model.parameters(), lr = cfg.clustering_lr)

    return loss_fn, encoder_optimizer, clustering_optimizer

class MaskedMSELoss(torch.nn.Module):

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = torch.nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)
    
class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature = 0.2):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(True)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def _cosine_simililarity(self, x, y):
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
    
class SCANLoss(torch.nn.Module):
    def __init__(self, entropy_weight):
        super(SCANLoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim = 1)
        self.bce = torch.nn.BCELoss()
        self.entropy_weight = entropy_weight

    def forward(self, anchors, neighbors):
        
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)
       
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)
        
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        total_loss = consistency_loss - self.entropy_weight * entropy_loss
        
        return total_loss

def entropy(x, input_as_probabilities, eps=1e-8):
    if input_as_probabilities:
        x_ =  torch.clamp(x, min = eps)
        b =  x_ * torch.log(x_)
    else:
        b = torch.functional.softmax(x, dim = 1) * torch.functional.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))
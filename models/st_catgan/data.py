from fddbenchmark import FDDDataloader
from scipy.signal import stft
import numpy as np
import torch

def _stft(ts, nperseg, noverlap):
    f, t, Zxx = stft(ts.transpose(0, 2, 1), nperseg=nperseg, noverlap=noverlap, boundary='even')
    Zxx = np.abs(Zxx)
    _min, _max = Zxx.min(axis=(1,2,3))[:, None, None, None], Zxx.max(axis=(1,2,3))[:, None, None, None]
    Zxx = (Zxx - _min) / (_max - _min) * 2 - 1
    return Zxx

class STFTDataloader(FDDDataloader):
    def __init__(self, nperseg, noverlap, in_cache, **kwargs):
        super().__init__(**kwargs)
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.in_cache = in_cache
        self.stft = dict()
    
    def __next__(self):
        ts, time_index, label = super().__next__()
        if self.in_cache:
            Zxx = self.stft.get(self.iter)
            if Zxx is None:
                Zxx = _stft(ts, self.nperseg, self.noverlap)
                self.stft[self.iter] = Zxx
        else:
            Zxx = _stft(ts, self.nperseg, self.noverlap)
        return torch.FloatTensor(Zxx), time_index, label
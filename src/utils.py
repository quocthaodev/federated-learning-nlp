import os
import random
import torch
import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_weights(model, init_type='xavier', init_gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    model.apply(init_func)

class MetricManager:
    def __init__(self, eval_metrics):
        self.results = {'loss': 0, 'accuracy': 0}
        self.count = 0
        self.correct = 0
    def track(self, loss, pred, true):
        self.count += len(true)
        self.results['loss'] += loss * len(true)
        correct = (pred.argmax(1) == true).sum().item()
        self.correct += correct
    def aggregate(self, total_len, curr_step=None):
        self.results['loss'] /= self.count
        self.results['accuracy'] = self.correct / self.count
        return self.results

class TqdmToLogger(tqdm):
    def __init__(self, *args, logger=None, mininterval=0.1, miniters=1, **kwargs):
        self._logger = logger
        super().__init__(*args, mininterval=mininterval, miniters=miniters, **kwargs)
    @property
    def logger(self):
        return self._logger if self._logger is not None else logging.getLogger(__name__)
    def write(self, x):
        if x.strip(): self.logger.info(x.strip())
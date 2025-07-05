import torch
import os

torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    device = 'cuda'
    map_location = None
else:
    device = 'cpu'
    map_location = torch.device('cpu')
    num_cores = os.cpu_count() - 2
    torch.set_num_threads(num_cores)
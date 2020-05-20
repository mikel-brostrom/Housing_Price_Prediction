import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics 
from itertools import chain 


def test(test_loader, model, device):
    model.eval()

    predictions = []
    labels = []
    print(len(test_loader))
    nb = len(test_loader)

    # Disable gradients
    pbar = tqdm(enumerate(test_loader), total=nb)
    for batch_idx, (sample, target) in pbar:
        with torch.no_grad():
            data = sample.to(device)
            pred = model(data.float()).detach().cpu().numpy()

            label = target.detach().cpu().numpy()

            predictions.append(pred)
            labels.append(label)

    # converting 2d list into 1d 
    # using chain.from_iterables 
    predictions = list(chain.from_iterable(predictions))
    labels = list(chain.from_iterable(labels)) 
    return metrics.r2_score(labels, predictions), metrics.mean_absolute_error(labels, predictions)


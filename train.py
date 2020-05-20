from sklearn.datasets import fetch_california_housing
from dataloader import HousesDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import Net
from torch.nn import MSELoss
import torch.cuda
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from test import test
from preprocessing import Preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


def train(device, crit, optimizer, model, train_loader):

    log_interval = 1000
    start_epoch = 0
    epochs = 150
    # the number of batches
    nb = train_loader.__len__()
    
    for epoch in range(start_epoch, epochs + 1):

        # ----------------------------------------------------------------
        # start epoch ----------------------------------------------------
        # ----------------------------------------------------------------

        print('\tEPOCH', epoch, '/', epochs)

        model.train()
        pbar = tqdm(enumerate(train_loader), total=nb)

        for batch_idx, (sample, target) in pbar:

            # ----------------------------------------------------------------
            # start batch ----------------------------------------------------
            # ----------------------------------------------------------------

            sample = sample.to(device)
            target = target.to(device)

            # zero the gradient buffers
            optimizer.zero_grad()

            # forward the data through the network
            output = model(sample.float())

            # calculate loss
            loss = crit(output.double(), target.double())

            # back propagate
            loss.backward()

            # update weights
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(sample), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

  

def main():


    # get house data
    houses = fetch_california_housing()

    x = houses.data
    y = houses.target

    df_data = pd.DataFrame(houses.data, columns=houses.feature_names)
    df_target = pd.DataFrame(houses.target, columns=['Target'])
    df_total = pd.concat([df_data, df_target], axis=1, sort=False)

    preprocessing = Preprocessing()
    df_total = preprocessing.standard_scaler(df_total)

    # split the data
    df_target = df_total.loc[:, df_total.columns == 'Target']
    df_data = df_total.loc[:, df_total.columns != 'Target']

    xtrain, xtest, ytrain, ytest = train_test_split(df_data, df_target, test_size=0.1, random_state=0)

    xtrain = xtrain.to_numpy()
    xtest = xtest.to_numpy()
    ytrain = ytrain.to_numpy()
    ytest = ytest.to_numpy()

    # create dataloaders for train and test
    dataset_train = HousesDataset(xtrain, ytrain)
    dataset_test = HousesDataset(xtest, ytest)

    train_loader = DataLoader(dataset_train,
                            batch_size=512,
                            num_workers=1,
                            shuffle=True,
                            pin_memory=True)
    test_loader = DataLoader(dataset_test,
                            batch_size=512,
                            num_workers=1,
                            shuffle=True,
                            pin_memory=True)

    # train and evaluate NN
    lr = 0.001
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    crit = MSELoss()

    train(device, crit, optimizer, model, train_loader)
    r2, MAE = test(test_loader, model, device)
    print('\tNN')
    print('r2 score:', r2)
    print('MAE score:', MAE)

    # train and evaluate other models
    regressor = LinearRegression()
    regressor.fit(xtrain, ytrain)
    preds = regressor.predict(xtest)
    print('\tLinear regressor')
    print('r2 score:', metrics.r2_score(ytest, preds))
    print('MAE score:', metrics.mean_absolute_error(ytest, preds))
    
    
    rf = RandomForestRegressor(random_state=42)
    rf.fit(xtrain, ytrain)
    preds = rf.predict(xtest)
    print('\tRandom forest')
    print('r2 score:', metrics.r2_score(ytest, preds))
    print('MAE score:', metrics.mean_absolute_error(ytest, preds))






if __name__ == "__main__":
    main()

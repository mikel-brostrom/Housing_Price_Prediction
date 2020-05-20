from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.special import boxcox1p
import pandas as pd

class Preprocessing():

    def standard_scaler(self, X):
        x = StandardScaler().fit_transform(X)
        X = pd.DataFrame(x, columns=X.columns)
        return X

    def boxcox_transform(self, X,y=None):
        X['AveRooms']=X['AveRooms'].apply(lambda x: boxcox1p(x,0.25))
        X['AveBedrms']=X['AveBedrms'].apply(lambda x: boxcox1p(x,0.25))
        X['HouseAge']=X['HouseAge'].apply(lambda x: boxcox1p(x,0.25))
        X['Population']=X['Population'].apply(lambda x: boxcox1p(x,0.25))
        X['AveOccup']=X['AveOccup'].apply(lambda x: boxcox1p(x,0.25))
        X['Latitude']=X['Latitude'].apply(lambda x: boxcox1p(x,0.25))
        X['MedInc']=X['MedInc'].apply(lambda x: boxcox1p(x,0.25))
        # an offset is needed becouse the data is negative
        X['Longitude']=X['Longitude'].apply(lambda x: boxcox1p(x+125,0.25))
        X['Target']=X['Target'].apply(lambda x: boxcox1p(x,0.25))
        return X
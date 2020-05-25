# Housing_Price_Prediction

The idea of this project was to create a predictor on the california housing dataset. The data pertains to the houses found in a given California district and some summary stats about them based on the 1990 census data.

## The data

The data is comprised of 8 attributes

* MedInc median income in block

* HouseAge median house age in block

* AveRooms average number of rooms

* AveBedrms average number of bedrooms

* Population block population

* AveOccup average house occupancy

* Latitude house block latitude

* Longitude house block longitude

as well as the target, the housing price

## Training

`train.py` runs the training for three different models: NN, linear regression and random forest on the scikit-learn california housing dataset:

```bash
python3 train.py
```

Training output example:

```bash
...

Train Epoch: 150 [0/18576 (0%)] Loss: 0.130984
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 37/37 [00:00<00:00, 143.57it/s]
5
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 102.93it/s]
        NN
r2 score: 0.8232532827300671
MAE score: 0.2833885734455647
        Linear regressor
r2 score: 0.6098033978087847
MAE score: 0.4635741867691994
        Random forest
r2 score: 0.8138137169848451
MAE score: 0.2837869675577879
...
```

### The network

We use a stack fully connected layers with ReLU. The r2 score and MAE was used for evaluating the model 

## Conclusion

The neural network trained on the standardized signals gave the best model with an R2 score of 82.4. The models trained on the first two principal componentes gave a poor result even if they accounted for ~96% of the data variance.


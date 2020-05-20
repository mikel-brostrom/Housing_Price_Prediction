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

`train.py` runs the training on the sklearn california housing dataset:

```bash
python3 train.py
```

Training output example:

```bash
...

Train Epoch: 600 [0/18576 (0%)] Loss: 0.030689
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 37/37 [00:00<00:00, 155.59it/s]
37
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 37/37 [00:00<00:00, 159.08it/s]
r2 score: 0.9603513041286216
...
```

### The network

We use a stack fully connected layers with ReLU. The r2 score and MAE was used for evaluating the model 

## Conclusion

The random forest model trained on the boxcox tranformed signals gave the best model with an R2 score of 83. Not bad! The models trained on the first two principal componentes gave a poor result even if they accounted for ~96% of the data variance.


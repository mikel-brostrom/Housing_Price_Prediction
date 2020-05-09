# Housing_Price_Prediction

The idea of this project was to create a predictor on the california housing dataset. The data pertains to the houses found in a given California district and some summary stats about them based on the 1990 census data.

## The data

The data is comprised of 8 attrubutes

* MedInc median income in block

* HouseAge median house age in block

* AveRooms average number of rooms

* AveBedrms average number of bedrooms

* Population block population

* AveOccup average house occupancy

* Latitude house block latitude

* Longitude house block longitude

as well as the target, the housing price

## Conclusion

The random forest model trained on the boxcox tranformed signals gave the best model with an R2 score of 83. Not bad! The models trained on the first two principal componentes gave a poor result even if they accounted for ~96% of the data variance.

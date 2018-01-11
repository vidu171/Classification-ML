# Classification-ML
Application of Classification models using Python 3

Applying Classification on the dataset of the users of a social network who saw the advertiesment for a SUV and decided weather or not to buy the SUV.

| Age | Salary | Purchase|
|----------|---------------|----------------|
19|19000|0|
35|20000|0|
26|43000|0|

0 denotes that the user did not buy the SUV whereas 1 denotes that the user did bought the SUV


## K-Nearest Neighbour

K nearest Neighbour Classification Model works by classifing the values depending on the nearest Neighbour


#### Creating the Classifiation
With THe number of Neighbours as 5 and the power parameter of minkowski metric as 2 which means using the eucledian_distance

```python
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
```
#### Creating the confusion Matrix for To Check the Success of the Classification
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```
The Wrong Predictions calculated using the Confusion Matrix after Fitting the classification model is 7

#### Plot obtained 
![alt text](https://github.com/vidu171/Classification-ML/blob/master/K-NN/Figure_1.png "Graph with 0.01 resolution")



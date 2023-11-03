# CS 456 - Exam 2

Given the air pollution dataset, please use an MLP (multilayer perceptrons) model to predict the
level of pollutant PM2.5. Specifically, your MLP model needs: 

1. Two hidden layers with ReLU activation functions.
2. Use features SO2, PM10, CO, O3_8, airPressure, sunHours, highTemperature, lowHumidity, and season to predict PM2.5
3. Pre-process the dataset, such as checking if there is any missing value in the dataset, normalizing the continuous values (z-score), one-hot encoding discrete values.
4. Measure your model performance in terms of MSE (mean squared error) and MAE (mean absolute error).
5. Assume PM2.5 >= 30 is high (unhealthy). Please define a classifier to predict days that are healthy and unhealthy.

## 0. Development Environment Setup

#### 0.1 Setup Virtual Environment

So we are not installing python packages to our local machines, we can use the `venv` package to create a local environment.

To do so, run the command below

```
python3 -m venv ~/.virtualenvs/cs456
```

This creates a new virtual environment at location `~/.virtualenvs/cs456`. To access this virtual environment, run the command below

```
source ~/.virtualenvs/cs456/bin/activate
```

If this command does not work, try replacing `source` with `.`. The command will look like this

```
. ~/.virtualenvs/cs456/bin/activate
```

#### 0.2 Install Dependencies

Included in this repo is a `requirements.txt` file that will allow you to install all the necessary packages to run the code.
Install them with the following command

```
pip install -r requirements.txt
```

Once this is done, you can start up the project with the command 

```
python ex2.py
```

#### 0.2.1 Errors Installing Dependencies

For convenience, here are the packages used in the project.

- `pandas`
- `numpy`
- `torch`
- `scikit-learn`

When you install these, their dependencies will get installed and so on. So if you run the command `pip freeze`, you will see more than these four installed once all is done.

#### 0.3 Conclusion

Once you have reached this point, you should be able to run the project as described in `0.2`. 

## 1. Two Hidden Layers with ReLU activation functions

To accomplish this, we define a new MLP model with the following code snippet

```python
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x
```

## 2. Use features SO2, PM10, CO, O3_8, airPressure, sunHours, highTemperature, lowHumidity, and season to predict PM2.5

To accomplish this, we determine the features we would like to use, and we drop anything else from the data sets.

```python
# Continuous are numbers, and discrete will be one-hot encoded
# and the categories added as individual features with a binary value associated with them.

cont_features = ['SO2', 'PM10', 'CO', 'O3_8', 'airPressure', 'sunHours', 'highTemperature', 'lowHumidity']
disc_features = ['season']

# Our target is PM25

target = 'PM25'

X = data.drop(columns=[target, 'date', 'cityname', 'citycode', 'year', 'month', 'day', 'longitude', 'latitude'])
y = data[target]

```

## 3. Pre-process the dataset, such as checking if there is any missing value in the dataset, normalizing the continuous values (z-score), one-hot encoding discrete values.

We accomplish this with this code

```python
# We make sure to encode our data that way any categorical features
# are one-hot encoded and added as individual features.

data = pd.get_dummies(data, columns=disc_features)

# To normalize the data, we use the following code

scaler = StandardScaler()
data[cont_features] = scaler.fit_transform(data[cont_features])
```

## 4. Measure your model performance in terms of MSE (mean squared error) and MAE (mean absolute error).

After we trained our model, we can see how it performs!

We do so with this code

```python
# Calculate MSE and MAE
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
```

### 4.1 Results

```
Mean Squared Error (MSE): 95.52393341064453
Mean Absolute Error (MAE): 6.989352226257324
```

The mean squared error is actually not very good. Considering the average for PM25 is close to `35`, this error is HUGE. The mean absolute error is also not very good. It is almost `7`, which is a lot considering the average is `35`.

I would consider tweaking this NN to see if we can get better results in the future.

## 5. Assume PM2.5 >= 30 is high (unhealthy). Please define a classifier to predict days that are healthy and unhealthy.

To accomplish this, we can use the following code

```python
# Define a binary classifier
threshold = 30
y_pred_binary = (y_pred >= threshold).float()
y_test_binary = (y_test >= threshold).float()

# Calculate classification accuracy
accuracy = (y_pred_binary == y_test_binary).sum().item() / len(y_test)
print(f'Classification Accuracy: {accuracy}')
```

### 5.1 Results

```
Classification Accuracy: 0.8407589599437807
```

This accuracy is actually very good! It is `84%` accurate, which is very good. This means that our model is able to predict whether or not a day is unhealthy or healthy with `84%` accuracy. This is very good, and I am very happy with this result.
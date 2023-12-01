# CS 456 Project

Air pollution now ranks as the fourth most deadly health hazard globally, contributing to one out of every ten fatalities. Consequently, health and environmental bodies have elevated the mitigation of air pollution to their highest concern. Effective air pollution management is dependent upon prompt monitoring of air quality and precise forecasting of air pollutant variations, which include particulate matter (PM2.5 and PM10), carbon monoxide (CO), sulfur oxides, nitrogen oxides, and lead, among others.

For this project, you are tasked with the development and execution of a neural network model that predicts PM2.5 concentrations. The essential criteria for your model include:

- Employ historical data utilizing RNN-based neural networks, such as LSTM (Long Short-Term Memory) or GRU (Gated Recurrent Units).
- Incorporate the impact of neighboring cities using CNN-based (Convolutional Neural Network) neural networks.
- Assess your models using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), or accuracy, precision, and recall, based on how you define the problem. 
- Compare the performance of your model with benchmark methods. 
- Graphically represent the outcomes of your evaluations.

**Additional Tasks**

Additional tasks for your project involve a thorough analysis and interpretation of your model. Specifically, dissect the contributions of your neural network components to the predictions. Apply Shapley value techniques, such as SHAP, to elucidate the significance of the predictive features.

## Employ Historical Data utilizing RNN based neural networks

For this implementation I went with a LSTM model. My input layer has 21 features for the number of datapoints we are observing. There are 2 hidden layers with 64 neurons each. 

To measure the accuracy of the model, I used the mean squared error (MSE). It gives us an idea of how far off the model was in its predictions from the actual output. The lower the MSE, the better the model is performing.

To optimize the model, I used the Adam optimization with a loss rate of `1e-3`. I did not use a drop out rate.

To train the model, there are 10 epochs that will run. The batch size for the train and test loader is 64.

## Incorporate the impact of neighboring cities using CNN based neural networks

I added a CNN path to the LSTM model. The CNN path has 32 output channels, a kernel size of 3, and a stride of 1. The CNN is then joined with the LSTM path and given to the connected layer.

# Unfortunately...

I was unable to get the model to run. I kept getting an error that said 

```
     15         # LSTM path
     16         lstm_out, _ = self.lstm(x)
---> 17         lstm_out = lstm_out[:, -1, :]  # Take the output from the last time step
     18 
     19 

IndexError: too many indices for tensor of dimension 2
```

So, this is unfortunately all I am able to present for this project. I even took extra days past the due date to see if I could solve this error and I came up with nothing.

So you can see the error and exactly where it was haunting me, I have also migrated everything to a Google Colab notebook and it shows where it went wrong there. You can find that notebook ![here](https://colab.research.google.com/drive/1kddja6Dd24jMtklQRvGa4tshsyZhD1cN?usp=sharing)

## References

- https://d2l.ai/
- https://towardsdatascience.com/understanding-rnns-lstms-and-grus-ed62eb584d90
- https://www.tensorflow.org/tutorials/structured_data/time_series

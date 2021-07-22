# transfer-learning-soil-moisture-prediction
The skillful soil moisture (SM) for Soil Moisture Active Passive (SMAP) L4 product can provide substantial value for many practical applications including ecosystem management and precision agriculture. Deep learning (DL) models for hydrologic prediction provide a powerful method to build a concise prediction model of SM. However, the total number of daily SM samples in SMAP product is quite small, which may lead to overfitting and further impact the accuracy of DL models. From this, we first test whether the Convolutional Neural Networks, Long Short-Term Memory, and Convolutional LSTM models, which are frequent used for hydrologic prediction, can be reliably achieved excellent predictive performance. Then we pre-train the DL models based on the source domain (ERA5-land datasets) and fine-tune in target domain (SMAP dataset). The result shows that the transfer ConvLSTM model has the highest R2 ranging from 0.909 to 0.916, the lowest RMSE ranging from 0.0239 to 0.0247 in which predictions at the 3rd, 5th, and 7th days in the future and the lines between predicted SM and observed SM are closer to the ideal line (y = x) than all the other DL models. All the performance of transfer DL models are better than that of corresponding DL models without transfer learning. 


# Download the data
you can download SMAPdata from
https://smap.jpl.nasa.gov/data/
and ERA5data from
https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form 

# MODEL
The ERA5test  folder contains the model for predicting ERA5data
The SMAPtest folder contains the model for predicting SMAPdata
The fintune-SMAPtest folder contains the model for predicting SMAPdata,which is Pre-training on the ERA5data

the data-processing-XXX-model  can reshape the data from 3 variables which shape is [Sample size,1,height,width] to 
[Sample size,lead_time,3,height,width],3 mean precipitation ,lagged Soil moisture,soil temperature.


# Evaluation
You can find the evaluation metrics we used here in utils/evalution.py.
The document contains three evaluation metrics which are RMSE,MSE,R2.
The utils/Drawmap.py can draw theThe prediction map from our neural network and the Observation map

# uils
There are some Tooling code we made in this folder.


# Result Show
Here are some important results graphs

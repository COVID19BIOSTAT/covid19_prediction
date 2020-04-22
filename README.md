# Survival-Convolution Model for COVID-19 Prediction

<img src="https://img.shields.io/badge/Study%20Status-Results%20Available-yellow.svg" alt="Study Status: Results Available"> 

We present a parsimonious and robust survival-convolution model to predict daily new cases and latent cases. The model accounts for transmission during a incubation period and uses a time-varying reproduction number to reflect the temporal trend and change of transmission in response to an intervention. 

- Title: **Survival-Convolution Models for Predicting COVID-19 Cases and Assessing Effects of Mitigation Strategies** (medRxiv link: https://www.medrxiv.org/content/10.1101/2020.04.16.20067306v1)
- Authors: Qinxia Wang MPhil, Shanghong Xie PhD, Correspondonce to: Dr. Yuanjia Wang and Dr. Donglin Zeng
- Institutes: 
  + Department of Biostatistics, Mailman School of Public Health, Columbia University, New York, NY, USA 
  + Department of Biostatistics, Gillings School of Public Health, University of North Carolina at Chapal Hill, Chapal Hill, NC, USA



## Real Time Prediction (Updated on April 21, 2020)
### US Daily New Cases:

![](https://github.com/COVID19BIOSTAT/covid19_prediction/blob/master/example/US_fit.png)

### Italy Daily New Cases:

![Italy](https://github.com/COVID19BIOSTAT/covid19_prediction/blob/master/example/Italy_fit.png)

Note: training data used for model fitting is up to April 10, 2020. Dot: confirmed daily new cases; Red shade: 95% confidence interval of our prediction. 

## Setup Requirements

+ Python under version 3.7
+ Tensorflow under version 2.1.0 

## Example

### Input data
Suppose that we have an `numpy.array` `train_data` indicating observed daily confirmed/diagnosed cases starting from the first observed case. We will save it as "train_data.npy".
```
>>> train_data
array([    1.,    20.,     0.,     0.,    18.,     4.,     3.,     0.,
           3.,     5.,     7.,    25.,    24.,    34.,    63.,    98.,
         116.,   106.,   163.,   290.,   307.,   329.,   553.,   587.,
         843.,   983.,  1750.,  2950.,  4569.,  5632.,  4848.,  9400.,
       10311., 11166., 13451., 17388., 18743., 19452., 20065., 20732.,
       24914., 26655., 30107., 32454., 34196., 25400., 31240., 33502.,
       31997., 33606., 33752.], dtype=float32)
```

### Command
Now we want to fit this training data into our model.
We need to decide the following __arguments__.

+ `knots` - length of piecewise segments for the infection rate. E.g., with 81 training data, we decide a knot after 23 days, so `knots` will be `23,28`.
+ `input_path` - directory of the input `train_data.npy`.
+ `output_path` - directory to save output files.
+ `max_epochs` - number of epochs used in tensorflow optimizer (default: Adam), usually from 1000 to 3000 depending on numbers of knots and data size.
+ `initial_a` - initial values for the weights `a`. The length is `len(knots)+1`.
+ Other arguments can be changed including 
  + `learning_rate` (default: 0.01)
  + `test_duration` - number of days to predict (default: 100)
  + `min_t0`,`max_t0` - range of t0 to iterate (default: 1 to 21)

 Then we can use the following command to run the model in our example.

 `python run_single_machine_model.py --knots=23,28 --input_path="example/train_data.npy" --output_path="example/output" --max_epochs=2000 --initial_a=0.5,0.5,0.5 --learning_rate=0.01`

 The running time increases as number of knots, number of epochs and data size increases. The code can be adjusted for using multiple CPU cores to reduce the time, using the following command.

 `python run_multi_machine_model.py --knots=23,28 --input_path="example/train_data.npy" --output_path="example/output" --max_epochs=2000 --initial_a=0.5,0.5,0.5 --learning_rate=0.01`

### Output
After running the above command, we should have the following output files in `output_path`.
+ `best_t0.npy` - estimated t0
+ `best_weights.npy` - estimated `a`. E.g., in our example, the estimates for `a` contains three values. The infection rate is constant before the first observed case with value a1 and then linearly change to a2 during the first 23 days and from a2 to a3 in the next 28 days.
+ `predicted_infection_rate.npy` - infection rate from the first observed case.
+ `predicted_reproduction_number.npy` - reproduction number from the first observed case.
+ `predicted_daily_observed.npy` - daily new observed cases from the first observed case.
+ `predicted_daily_infected.npy` - cumulative latent cases on each day from the first observed case. 

Using the output files, we can visualize the observed and predicted daily new cases,

![daily new](https://github.com/COVID19BIOSTAT/covid19_prediction/blob/master/example/predicted.png)

as well as the piecewise infection rate and the reproduction number (Rt).

![infection rate](https://github.com/COVID19BIOSTAT/covid19_prediction/blob/master/example/infection.png)








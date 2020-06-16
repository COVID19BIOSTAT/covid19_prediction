# Survival-Convolution Model for COVID-19 Prediction

<img src="https://img.shields.io/badge/Study%20Status-Results%20Available-yellow.svg" alt="Study Status: Results Available"> 

The coronavirus disease COVID-19 has created major health crisis around the world. It is imperative to predict the disease epidemic, investigate the impacts of containment and mitigation measures on infection rates, and compare between countries. 
Existing methods for infectious disease modeling are SEIR models that rely on many untestable prior assumptions (e.g., fitting past influenza data) and unreliable with wide prediction intervals. 

We develop a robust **survival-convolution model** with few parameters that incorporates **the date of unknown patient zero, latent incubation periods**, and **time-varying reproduction numbers**. 


- Title: **Survival-Convolution Models for Predicting COVID-19 Cases and Assessing Effects of Mitigation Strategies** 
<br/> manuscript: https://github.com/COVID19BIOSTAT/covid19_prediction/blob/master/manuscript/COVID_Final.pdf
<br/> medRxiv link: https://www.medrxiv.org/content/10.1101/2020.04.16.20067306v1

- Authors: **Qinxia Wang<sup>a</sup>, Shanghong Xie<sup>a</sup>, Yuanjia Wang<sup>a</sup>, and Donglin Zeng<sup>b</sup>**
- Institutes: 
  + 1. **Department of Biostatistics, Mailman School of Public Health, Columbia University, New York, NY, USA**
  + 2. **Department of Biostatistics, Gillings School of Public Health, University of North Carolina at Chapal Hill, Chapal Hill, NC, USA**
- Correspondonce to: **Dr. Yuanjia Wang (yw2016@cumc.columbia.edu) and Dr. Donglin Zeng (dzeng@email.unc.edu)**



## Real Time Prediction
Data source: [Worldmeters](https://www.worldometers.info/coronavirus/)

Note: Once the testing capacity is increased, the trend will change again. These are beyond what our model can predict. Since April 14 2020, CDC case counts include both confirmed and probable cases following [new CDC guidelines](https://www.worldometers.info/coronavirus/us-data/). 

### US Daily New Cases (with training data up to May 29):

![](https://github.com/COVID19BIOSTAT/covid19_prediction/blob/master/example/US_fit_update.png)

Observed and predicted daily new cases, with a 95% prediction interval.

First dashed line indicates the declaration of national emergency (Mar 13). Second to fourth dashed lines indicate knots with interval of two weeks (Mar 27, Apr 10, Apr 24). Training data: February 21 to May 29; Testing data: May 30 to date.

### US Daily New Cases (with training data up to May 1):

![](https://github.com/COVID19BIOSTAT/covid19_prediction/blob/master/example/US_fit_intervention.png)

Observed and predicted daily new cases, 95% prediction intervals under four scenarios.

Scenario 1: infection rate a(t) follows the same trend after May 1 as observed between March 27 and May 1. 

Scenario 2: rate of decrease of a(t) slows by 50% after May 1.

Scenario 3: rate of decrease of a(t) slows by 75% after May 1. 

Scenario 4: rate of decrease of a(t) slows by 100% after May 1.

First dashed line indicates the declaration of national emergency (March 13). Second dashed line indicates two weeks after (Mar 27). Training data: Feb 21 to May 1; Testing data: May 2 to date.


### US Reproduction Number (with training data up to May 1):

![](https://github.com/COVID19BIOSTAT/covid19_prediction/blob/master/example/US_R0_intervention.png)


### Italy Daily New Cases:

![Italy](https://github.com/COVID19BIOSTAT/covid19_prediction/blob/master/example/Italy_fit.png)

Observed and predicted daily new cases and 95% prediction interval (shaded). First dashed line indicates the nation-wide lockdown (Mar 11). Second and third dashed line indicates two or four weeks after. Training data: Feb 20 to Apr 29; Testing data: Apr 30 to date.



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






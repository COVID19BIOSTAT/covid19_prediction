# Survival-Convolution Model for COVID-19 Prediction

<img src="https://img.shields.io/badge/Study%20Status-Results%20Available-yellow.svg" alt="Study Status: Results Available"> 

The coronavirus disease COVID-19 has created major health crisis around the world. It is imperative to predict the disease epidemic, investigate the impacts of containment and mitigation measures on infection rates, and compare between countries. 
Existing methods for infectious disease modeling are SEIR models that rely on many untestable prior assumptions (e.g., fitting past influenza data) and unreliable with wide prediction intervals. 

We develop a robust **survival-convolution model** with few parameters that incorporates **the date of unknown patient zero, latent incubation periods**, and **time-varying reproduction numbers**. 


- Title: **Survival-Convolution Models for Predicting COVID-19 Cases and Assessing Effects of Mitigation Strategies** 
- Authors: **Qinxia Wang<sup>a</sup>, Shanghong Xie<sup>a</sup>, Yuanjia Wang<sup>a</sup>, and Donglin Zeng<sup>b</sup>**
- Institutes: 
  + 1. **Department of Biostatistics, Mailman School of Public Health, Columbia University, New York, NY, USA**
  + 2. **Department of Biostatistics, Gillings School of Public Health, University of North Carolina at Chapal Hill, Chapal Hill, NC, USA**
- Correspondonce to: **Dr. Yuanjia Wang (yw2016@cumc.columbia.edu) and Dr. Donglin Zeng (dzeng@email.unc.edu)**
- [Manuscript](https://www.frontiersin.org/articles/10.3389/fpubh.2020.00325/full?&utm_source=Email_to_authors_&utm_medium=Email&utm_content=T1_11.5e1_author&utm_campaign=Email_publication&field=&journalName=Frontiers_in_Public_Health&id=561170): Wang Q, Xie S, Wang Y, and Zeng D (2020). Survival-Convolution Models for Predicting COVID-19 Cases and Assessing Effects of Mitigation Strategies. Frontiers in Public Health 8(2020) 325. 

Our model is also used by CDC for [COVID-19 ensemble forecast](https://www.cdc.gov/coronavirus/2019-ncov/covid-data/forecasting-us.html).


## Real Time Prediction

Note: Once the testing capacity is increased, the trend will change again. These are beyond what our model can predict. Since April 14 2020, CDC case counts include both confirmed and probable cases following [new CDC guidelines](https://www.worldometers.info/coronavirus/us-data/). 

Data source for US: [JHU CSSE group](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series)
### US Daily New Cases (with training data up to August 21):

![](https://github.com/COVID19BIOSTAT/covid19_prediction/blob/master/example/plot817/daily_case.png)

Observed and predicted daily new cases, with a 95% prediction interval.

First dashed line indicates the declaration of national emergency (Mar 13). Second to seventh dashed lines indicate knots with interval of two or three weeks (Mar 27, Apr 10, May 1, May 22, June 26). Training data: February 21 to August 14; Test data: August 22.

### US Cumulative Deaths (with training data up to August 21):
![](https://github.com/COVID19BIOSTAT/covid19_prediction/blob/master/example/plot817/cumulative_death.png)

Observed and predicted cumulative deaths, with a 95% prediction interval. First to third dashed lines indicate knots at May 1, May 22, June 26 account for reopen. 

### US Daily Inc Deaths (with training data up to August 21):
![](https://github.com/COVID19BIOSTAT/covid19_prediction/blob/master/example/plot817/daily_death.png)

Observed and predicted daily deaths, with a 95% prediction interval. First to third dashed lines indicate knots at May 1, May 22, June 26 account for reopen. 




Data source for Italy: [Worldmeters](https://www.worldometers.info/coronavirus/)

### Italy Daily New Cases (with training data up to April 29)::

![Italy](https://github.com/COVID19BIOSTAT/covid19_prediction/blob/master/example/Italy_fit_rev_080820.png)

Observed and predicted daily new cases and 95% prediction interval (shaded). First dashed line indicates the nation-wide lockdown (Mar 11). Second and third dashed line indicates two or four weeks after. Training data: Feb 20 to Apr 29; Testing data: Apr 30 to August 7.



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







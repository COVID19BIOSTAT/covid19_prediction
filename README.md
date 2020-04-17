# Survival-Convolution Model for COVID-19 Prediction

We present a parsimonious and robust survival-convolution model to predict daily new cases and latent cases. The model accounts for transmission during a incubation period and uses a time-varying reproduction number to reflect the temporal trend and change of transmission in response to an intervention. 

## Setup requirements

+ Python under version 3.7
+ Tensorflow under version 2.1.0
+ Other basic libraries including Numpy, Scikit-learn, etc. 

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

+ `knots` - length of segments of pieces for the infection rate. E.g., with 81 training data, we decide a knot after 23 days, so `knots` will be `23,28`.
+ `input_path` - directory of the input `train_data.npy`.
+ `output_path` - directory to save output files.
+ `max_epochs` - number of epochs used in tensorflow optimizer (default: Adam), usually from 1000 to 3000 depending on numbers of knots and data size.
+ `initial_a` - initial values for the weights `a`. The length is `len(knots)+1`.
+ Other arguments can be changed including 
  + `learning_rate` (default: 0.01)
  + `test_duration` - number of days to predict (default: 100)
  + `min_t0`,`max_t0` - range of t0 to iterate (default: 1 to 21)

 Then we can use the following command to run the model in our example.

 `python run_single_machine_model.py --knots=23,28 --input_path="example/train_data.npy" --output_path="example/output" --max_epoch=2000 --initial_a=0.5,0.5,0.5 --learning_rate=0.01`

 The running time increases as number of knots, number of epochs and data size increases. The code can be adjusted for using multiple CPU cores to reduce the time. 

### Output files




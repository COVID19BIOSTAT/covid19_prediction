"""Python module for COVID-19 Survival-Convolution Combined Model.

In this module we trains an epidemic model for COVID-19 based on the number of
symptomatic cases and the number of deaths and recoveries.
"""

import piecewise_linear_infection_model as infection_model
import death_model
from typing import Optional, Sequence, Text, Tuple
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')


class Covid19CombinedPredictModel(death_model.Covid19DeathPredictModel):
  """The combined epidemic model for Covid-19.

  In this model, we will use the number of daily new confirmed cases, and
  daliy new death tolls as training data. We will learn import features about
  this disease such as basic reproduction number and death rate, and will
  predict the number of infections and the number of death tolls in the near
  future.
  """

  def __init__(self, **kwargs):
    """Initializes an instance of Covid19CombinedPredictModel."""
    super(Covid19CombinedPredictModel, self).__init__(**kwargs)

  def daily_combined(self, inputs: np.ndarray,
                     static_tensorshape: bool = False):
    """Returns the number of daily new death cases in an array.

    Args:
      inputs: a reformulated sequence of predictors that fits tensorflow model.
        For the detailed formats, see method "_get_trainable_x()" in class
        Covid19InfectionsEstimator.
      static_tensorshape: if True, this method will return tensor with a fixed
        length in training a model. Otherwise, the length can be any positive
        integer, which is used in predicting the number of new confirmed cases
        in future.

    Raises:
      xxx

    Returns:
      A 1d tensor for storing the number of daily new death cases. The length
      is same as the length of argument "inputs".

    """
    return tf.transpose(
        tf.stack(
            [self.daily_observed(inputs, static_tensorshape),
             self.daily_death(inputs, static_tensorshape)]
        ))

  def call(self, inputs: np.ndarray) -> infection_model.TensorType:
    """Returns the number of confirmed cases and deaths on a daily basis.

    This method needs to be overriden to subclass keras.Model.

    Args:
      inputs: a reformulated sequence of predictors that fits tensorflow model.
        For the detailed formats, see method "_get_trainable_x()" in class
        Covid19InfectionsEstimator.

    Returns:
      A 2d tensor for storing the number of daily new death cases. The first
      shape (row) is same as the length of argument "inputs". Within each row,
      two elements are the number of confirmed cases and number of deaths on
      that day.

    """
    return self.daily_combined(inputs, True)


class Covid19CombinedEstimator(death_model.Covid19DeathEstimator):
  """Selects the best combined model to predict Covid-19 cases and death tolls.

  xxx
  """

  def __init__(self, loss_prop: np.float64 = .9, **kwargs):
    """Initializes a Covid19CombinedEstimator instance.

    Args:
      loss_prop:

    """
    self._loss_prop = loss_prop
    super(Covid19CombinedEstimator, self).__init__(**kwargs)

  def _fit_with_t0(self,
                   data: Sequence[int], t0: int, message: Text,
                   enable_tensorboard: bool = False,
                   tensorboard_logdir: Optional[Text] = None
                   ) -> Tuple[
                       Covid19CombinedPredictModel,
                       infection_model.TensorType]:
    """Returns the best combined model after training with a given t0.

    Args:
      data: training data (number of daily new cases and death tolls) in a 2d
        array.
      t0: specifies the number of days between the occurrence of the first
        infected case (patient zero) and the first observed case.
      message: optionally pass a prefix string in the filenames of training
        weights (in the format of hdf5 file). We will generate a lot of such
        files in the training process.
      enable_tensorboard: whether or not use tensorboard to monitor training.
      tensorboard_logdir: xxx.


    Returns:
      model: the best model after training with t0.
      loss: the loss of the best model after training with t0.

    """
    model = Covid19CombinedPredictModel(
        n_weights=2 * len(self._knots) + 1 - sum(self._knots_connect),
        t0=t0,
        len_inputs=len(data) + t0,
        max_latency=self._estimator_args.get(
            "max_latency", infection_model.DEFAULT_MAX_LATENCY),
        max_intervention=self._estimator_args.get(
            "max_intervention", death_model.DEFAULT_MAX_INTERVENTION),
        **self._model_args)

    x = self._get_trainable_x(len(data), t0)
    # Pad t0 elements at front to be 0.
    y = np.pad(data, [(t0, 0), (0, 0)]).astype(np.float64)

    # Define the loss function for each t0 value. Compare the square-root
    # difference.
    def custom_loss(y_actual, y_pred):
      loss_case = self._estimator_args.get(
          "loss_function", tf.keras.losses.MSE)(
              tf.math.sqrt(y_actual[t0:, 0]), tf.math.sqrt(y_pred[t0:, 0]))
      loss_death = self._estimator_args.get(
          "loss_function", tf.keras.losses.MSE)(
#               tf.math.sqrt(y_actual[t0:, 1]), tf.math.sqrt(y_pred[t0:, 1]))
              (y_actual[t0:, 1]), (y_pred[t0:, 1]))
      return (1. - self._loss_prop) * loss_case + self._loss_prop * loss_death

    optimizer_option = self._estimator_args.get(
        "optimizer", tf.keras.optimizers.Adam)
    optimizer = optimizer_option(
        learning_rate=self._estimator_args.get("learning_rate", 0.01),
        clipnorm=1.0)
    model.compile(optimizer, custom_loss)
    callbacks, min_loss_filepath = Covid19CombinedEstimator._setup_callbacks(
        message, t0, enable_tensorboard, tensorboard_logdir)

    model.fit(
        x, y, epochs=self._estimator_args.get("epochs", 100),
        batch_size=len(data) + t0, shuffle=False,
        verbose=self._estimator_args.get("verbose", 0),
        callbacks=callbacks)

    model.load_weights(min_loss_filepath)
    loss = custom_loss(y, model(x))
    return model, loss

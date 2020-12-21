"""Python module for COVID-19 Survival-Convolution Death Model.

In this module we focus on predicting the number of deaths with a similar
approach to modeling the number of infections.
"""
from typing import Optional, Sequence, Text, Tuple
import numpy as np
import tensorflow as tf
import piecewise_linear_infection_model as infection_model
tf.keras.backend.set_floatx('float64')

AVE_DAYS_TO_OUTCOME = 18.6
DEFAULT_MAX_INTERVENTION = 40


class Covid19DeathPredictModel(infection_model.Covid19InfectionsPredictModel):
  """Model to predict the number of cases dying from Covid-19.

  Attributes:
    w: hazard rate parameter used in defining the survival function that
      specifies the average of time until an infected patient becoming
      symptomatic and being diagnosed positive.

    death_rate: a variable that indicates probability of death given infection.

  """

  def __init__(self,
               max_intervention: int = DEFAULT_MAX_INTERVENTION,
               **kwargs):
    """Initializes an instance of Covid19DeathPredictModel."""
    self._max_intervention = max_intervention
    self.w = tf.Variable(
        kwargs.pop("initial_guess_w", AVE_DAYS_TO_OUTCOME),
        name="w",
        dtype=tf.float64,
        trainable=kwargs.pop("variable_w_trainable", True))
    trainable_death_rate = kwargs.pop("variable_death_rate_trainable", True)
    initial_guess_death_rate = kwargs.pop("initial_guess_death_rate", 0.04)
    super(Covid19DeathPredictModel, self).__init__(**kwargs)
    self.death_rate = tf.Variable(
        np.ones([self._n_weights, 1]) * initial_guess_death_rate,
        name="death_rate",
        dtype=tf.float64,
        trainable=trainable_death_rate)

  def get_outcome_survival_probs(self):
    """Returns survival probabilities of time from onset to final outcome."""
    surv = tf.math.exp(
        tf.multiply(
            - 1. / self.w,
            tf.range(self._max_intervention, dtype=tf.float64)))
    return (surv - surv[self._max_intervention - 1]) / (
        1. - surv[self._max_intervention - 1])

  def get_death_conv_weights(self):
    """Returns the convolutional weights for modeling daily death numbers.

    These are the probability mass functions of the time from a case being
    infected until observing the final outcome (recovery / death). They are
    based on the assumption that: the incubation period, and the time between
    infection and final outcome (death / recovery) are independent.
    """
    symptomatic_surv = self.get_symptomatic_survival_probs()
    symptomatic_pmf = symptomatic_surv[:-1] - symptomatic_surv[1:]
    outcome_surv = self.get_outcome_survival_probs()
    outcome_pmf = outcome_surv[:-1] - outcome_surv[1:]

    return tf.squeeze(
        tf.nn.conv1d(
            tf.reshape(
                tf.pad(
                    outcome_pmf,
                    paddings=tf.constant(
                        [[self._max_latency, self._max_latency - 1]]
                    ),
                    mode="CONSTANT"),
                [1, 2 * self._max_latency + self._max_intervention - 2, 1]
            ),
            tf.reshape(
                tf.pad(
                    symptomatic_pmf[::-1],
                    paddings=tf.constant([[0, 1]]),
                    mode="CONSTANT"
                ),
                [self._max_latency, 1, 1]
            ), 1, "VALID"))

  def daily_death(self, inputs: np.ndarray,
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
    inputs_len = self._len_inputs if static_tensorshape else inputs.shape[0]
    daily_infected_cases = tf.reshape(
        tf.slice(
            self._daily_infected(inputs),
            begin=[0, self._max_latency - 1],
            size=[inputs_len, 1]
        ), [inputs_len])

    conv_weights_len = self._max_latency + self._max_intervention - 1
    conv_weights = self.get_death_conv_weights()
    if conv_weights.shape[0] != conv_weights_len:
      raise ValueError(
          f"The length of convolutional weights {conv_weights.shape[0]} is "
          f"different from the expected value {conv_weights_len}.")
    return tf.squeeze(
        tf.nn.conv1d(
            tf.reshape(
                tf.pad(
                    daily_infected_cases,
                    paddings=tf.constant(
                        [[conv_weights_len - 1, 0]]
                    ),
                    mode="CONSTANT"),
                [1, conv_weights_len + inputs_len - 1, 1]
            ),
            tf.reshape(
                conv_weights[::-1],
                [conv_weights_len, 1, 1]
            ), 1, "VALID")) * tf.reshape(  # Allow death rate change over time.
                tf.matmul(inputs, self.death_rate), [inputs_len])

  def call(self, inputs: np.ndarray) -> infection_model.TensorType:
    """Returns the number of deaths on a daily basis.

    This method needs to be overriden to subclass keras.Model.

    Args:
      inputs: a reformulated sequence of predictors that fits tensorflow model.
        For the detailed formats, see method "_get_trainable_x()" in class
        Covid19InfectionsEstimator.

    Returns:
      A 1d tensor for storing the number of daily new death cases. The length
      is same as the length of argument "inputs".

    """
    return self.daily_death(inputs, True)


class Covid19DeathEstimator(infection_model.Covid19InfectionsEstimator):
  """Selects the best model to predict Covid-19 death tolls."""

  def __init__(self, **kwargs):
    """Initializes a Covid19DeathEstimator instance."""
    super(Covid19DeathEstimator, self).__init__(**kwargs)

  def _fit_with_t0(self, data: Sequence[int], t0: int, message: Text,
                   enable_tensorboard: bool = False,
                   tensorboard_logdir: Optional[Text] = None
                   ) -> Tuple[
                       Covid19DeathPredictModel, infection_model.TensorType]:
    """Returns the death toll model after training with a given t0.

    Args:
      data: training data (number of daily new death tolls) in a 1d array.
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
    model = Covid19DeathPredictModel(
        n_weights=2 * len(self._knots) + 1 - sum(self._knots_connect),
        t0=t0,
        len_inputs=len(data) + t0,
        max_latency=self._estimator_args.get(
            "max_latency", infection_model.DEFAULT_MAX_LATENCY),
        max_intervention=self._estimator_args.get(
            "max_intervention", DEFAULT_MAX_INTERVENTION),
        **self._model_args)

    x = self._get_trainable_x(len(data), t0)
    # Pad t0 elements at front to be 0.
    y = np.pad(data, [t0, 0]).astype(np.float64)

    # Define the loss function for each t0 value. Compare the square-root
    # difference.
    def custom_loss(y_actual, y_pred):
      return self._estimator_args.get(
          "loss_function", tf.keras.losses.MSE)(
#               tf.math.sqrt(y_actual[t0:]), tf.math.sqrt(y_pred[t0:]))
              (y_actual[t0:]), (y_pred[t0:]))
    optimizer_option = self._estimator_args.get(
        "optimizer", tf.keras.optimizers.Adam)
    optimizer = optimizer_option(
        learning_rate=self._estimator_args.get("learning_rate", 0.01),
        clipnorm=1.0)
    model.compile(optimizer, custom_loss)
    callbacks, min_loss_filepath = Covid19DeathEstimator._setup_callbacks(
        message, t0, enable_tensorboard, tensorboard_logdir)

    model.fit(
        x, y, epochs=self._estimator_args.get("epochs", 100),
        batch_size=len(data) + t0, shuffle=False,
        verbose=self._estimator_args.get("verbose", 0),
        callbacks=callbacks)

    model.load_weights(min_loss_filepath)
    loss = custom_loss(y, model(x))
    return model, loss

  def predict_death(self, duration: int,
                    flatten_future: bool = False) -> Optional[
                        infection_model.TensorType]:
    """Predicts the number of new death cases reported on each day.

    Args:
      duration: specifies the number of days for prediction.
      flatten_future: this parameter takes effect in prediction only,
        indicating whether the infection rate or death rate is flattened in
        the future.

    Returns:
      The number of daily new death cases in 1d tensor. The length is equal to
      the value of duration.

    """
    if self._final_model is None:
      return None
    x_pred = self._get_trainable_x(
        duration, self._final_model.t0, flatten_future)
    return self._final_model.daily_death(x_pred)

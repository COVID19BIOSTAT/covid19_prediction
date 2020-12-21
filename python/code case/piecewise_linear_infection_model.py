"""Python module for COVID-19 Survival-Convolution Infection Model.

In this module we focus on modeling the number of infected cases, and the
number of symptomatic cases on a daily basis.
"""
from datetime import datetime
import numpy as np
import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Text, Tuple
import sklearn
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TerminateOnNaN
from tensorflow.keras.callbacks import TensorBoard
tf.keras.backend.set_floatx('float64')

DEFAULT_MAX_LATENCY = 21
AVE_DAYS_BEFORE_ONSET = 5.2
TensorType = tf.Tensor


class Covid19InfectionsPredictModel(tf.keras.Model):
  """A unified model to predicts the number of COVID-19 positive cases.

  Attributes:
    h: specifies inverse hazard rate used in defining the survival function.
      This value equals the average of time until an infected patient becoming
      symptomatic and being diagnosed positive.

   a: vector that indicates the infection rate at each knot. The length equals
     the length of 'knots' + 1.

   t0: number of days between the occurrence of the first infected case (the
     patient-zero) and the first observed case.

  """

  def __init__(self, n_weights: int, t0: int, len_inputs: int,
               max_latency: int = DEFAULT_MAX_LATENCY, **kwargs):
    """Initializes an instance of Covid19InfectionsPredictModel object.

    Args:
      n_weights: specifies the number of parameters we need for defining the
        infection rates. The value is the same as the length of knots (see
        class Covid19InfectionsEstimator) + 1 + the number of discontinous
        points.

      t0: specifies the number of days between the occurrence of the first
        infected case (patient zero) and the first observed case.

      len_inputs: equals the sum of t0 and length of the observed training
        data.

      max_latency: the maximum number of days for any infected patient to be
        asymptomatic.

      **kwargs: pass additional arguments to the model.

    """
    self._n_weights = n_weights
    self._t0 = t0
    self._len_inputs = len_inputs
    self._max_latency = max_latency
    self._daily_infected_initializer = tf.constant(
        np.append(np.zeros(self._max_latency - 1), 1), dtype=tf.float64)

    self.h = tf.Variable(
        kwargs.pop("initial_guess_h", AVE_DAYS_BEFORE_ONSET),
        name="h",
        dtype=tf.float64,
        trainable=kwargs.pop("variable_h_trainable", False))
    # The vector "a" indicates the infection rate at each knot. For example,
    # when the number of pieces is 2, then a = [a0, a1, a2] where a0 means the
    # infection rate before observing the first positive case; a0->a1 means
    # the change of infection rate in the first piece; and a1->a2 means the
    # change of infection rate in the second piece.
    self.a = tf.Variable(
        kwargs.pop(
            "initial_guess_a",
            0.5 * np.ones([self._n_weights])),
        name="a",
        dtype=tf.float64,
        trainable=kwargs.pop("variable_a_trainable", True))
    super(Covid19InfectionsPredictModel, self).__init__(**kwargs)

  @property
  def t0(self) -> int:
    """Estimates time between the patient zero and the first observed case."""
    return self._t0

  def load_pretrained_weights(self, weights: List[tf.Variable]):
    """Resets the model with external pretrained weights."""
    parsed_weights = {w.name: w.numpy() for w in weights}
    self.set_weights([parsed_weights[w.name] for w in self.weights])

  def get_symptomatic_survival_probs(self) -> TensorType:
    """Returns survival probabilities of being symptomatic after infection."""
    surv = tf.math.exp(
        tf.multiply(
            - 1. / self.h,
            tf.range(self._max_latency, dtype=tf.float64)))
    return (surv - surv[self._max_latency - 1]) / (
        1. - surv[self._max_latency - 1])

  def _daily_infected(self, inputs: np.ndarray) -> TensorType:
    """Returns the number of daily new infected cases in a matrix.

    Args:
      inputs: a reformulated sequence of predictors that fits tensorflow model.
        For the detailed formats, see method "_get_trainable_x()" in class
        Covid19InfectionsEstimator.

    Returns:
      A 2d tensor for computing the number of daily new infections in an
      iterative manner. The first shape is same as the length of argument
      "inputs" and the second shape equals "max_latency" defined by model.

    """
    reverse_surv = self.get_symptomatic_survival_probs()[::-1]

    # Define a helper function that generates the model-based value of new
    # infections on today based on past history of daily new infections. In
    # this model we adopt a survival-rate convolution model.
    def _evolve_daily_infected(input_a, input_b):
      past_daily_infected = tf.slice(
          input_a, begin=[1], size=[self._max_latency - 1])
      return tf.concat(
          [past_daily_infected,
           tf.reshape(
               tf.nn.relu(tf.tensordot(
                   self.a, input_b, 1)) * tf.tensordot(
                       past_daily_infected, reverse_surv[:-1], 1), [1])], 0)
    return tf.concat([
        tf.reshape(self._daily_infected_initializer,
                   [1, self._max_latency]),
        tf.scan(fn=_evolve_daily_infected, elems=inputs,
                initializer=self._daily_infected_initializer)[:-1]], 0)

  def daily_infected(self, inputs: np.ndarray) -> TensorType:
    """Returns the number of new infected cases per day in a sequence.

    Args:
      inputs: a reformulated sequence of predictors that fits tensorflow model.
        For the detailed formats, see method "_get_trainable_x()" in class
        Covid19InfectionsEstimator.

    Returns:
      A 1d tensor for storing the number of daily new infections. The length is
      same as the length of argument "inputs".

    """
    return tf.reshape(
        tf.slice(
            self._daily_infected(inputs),
            begin=[0, self._max_latency - 1], size=[inputs.shape[0], 1]),
        [inputs.shape[0]])

  def daily_observed(self, inputs: np.ndarray,
                     static_tensorshape: bool = False):
    """Returns the number of daily new observed (confirmed) cases in an array.

    Args:
      inputs: a reformulated sequence of predictors that fits tensorflow model.
        For the detailed formats, see method "_get_trainable_x()" in class
        Covid19InfectionsEstimator.
      static_tensorshape: if True, this method will return tensor with a fixed
        length in training a model. Otherwise, the length can be any positive
        integer, which is used in predicting the number of new confirmed cases
        in future.

    Returns:
      A 1d tensor for storing the number of daily new infections. The length is
      same as the length of argument "inputs".

    """
    inputs_len = self._len_inputs if static_tensorshape else inputs.shape[0]
    reverse_surv = self.get_symptomatic_survival_probs()[::-1]
    return tf.reshape(
        tf.matmul(
            tf.slice(
                self._daily_infected(inputs), begin=[0, 1],
                size=[inputs_len, self._max_latency - 1]
            ),
            tf.reshape(reverse_surv[1:] - reverse_surv[:-1],
                       [self._max_latency - 1, 1])
        ), [inputs_len])

  def infection_rate(self, inputs: np.ndarray) -> TensorType:
    """Estimates time-varying infection rates."""
    return tf.nn.relu(tf.reshape(
        tf.matmul(inputs, tf.reshape(self.a, [self._n_weights, 1])),
        [inputs.shape[0]]))

  def reproduction_number(self, inputs: np.ndarray) -> TensorType:
    """Estimates time-varying basic reproduction numbers."""
    return tf.squeeze(
        tf.nn.conv1d(
            tf.reshape(
                self.infection_rate(inputs),
                [1, inputs.shape[0], 1]),
            tf.reshape(
                self.get_symptomatic_survival_probs(),
                [self._max_latency, 1, 1]),
            1, "VALID"))

  def call(self, inputs: np.ndarray) -> TensorType:
    """Returns the number of new confirmed cases per day.

    This method needs to be overriden to subclass keras.Model.

    Args:
      inputs: a reformulated sequence of predictors that fits tensorflow model.
        For the detailed formats, see method "_get_trainable_x()" in class
        Covid19InfectionsEstimator.

    Returns:
      A 1d tensor for storing the number of daily new confirmed cases. The
      length is same as the length of argument "inputs".

    """
    return self.daily_observed(inputs, True)


class Covid19InfectionsEstimator(sklearn.base.BaseEstimator):
  """Selects the best epidemic model based on observable new cases.

  Attributes:
    final_model: the best Covid19InfectionsPredictModel after training on the
      history of daily new confirmed cases.
    final_loss: the value of loss function given the final model we obtain
      after training.

  """

  def __init__(self, knots: List[int], knots_connect: List[int],
               estimator_args: Dict[Text, Any], **kwargs):
    """Initializes a Covid19CaseEstimator instance.

    Args:
      knots: a list of integers in which each represents the length of one
        piece in the piecewise linear infection rate model. These integers
        should sum up to the length of training data.
      knots_connect: a list of integers in which each represents whether we
        allow the piece to be continous at the beginning (1: continous and 0:
        discontinous).
      estimator_args: change to non-default arguments for training this
        estimator. User may change: the choice of optimizer, learning rate,
        loss function, and the number of training epochs.
      **kwargs: pass additional arguments to the estimator. For example, the
        initial values in a Covid19InfectionsPredictModel.

    Raises:
      ValueError: if any number in "knots" is zero or negative, or the lengths
        of 'knots' and 'knots_connect' are not equal.

    """
    self._knots = knots
    if min(self._knots) <= 0:
      raise ValueError("The elements in vector 'knots' must be positive.")
    self._knots_connect = knots_connect
    if len(self._knots) != len(self._knots_connect):
      raise ValueError("The length of 'knots_connect' must be the same as the"
                       " length of 'knots'.")
    self._estimator_args = estimator_args
    self._model_args = kwargs
    self._final_model = None
    self._final_loss = None
    super(Covid19InfectionsEstimator, self).__init__()

  @property
  def final_model(self) -> Optional[Covid19InfectionsPredictModel]:
    """Returns the final model after training with different t0."""
    return self._final_model

  @property
  def final_loss(self) -> Optional[np.float64]:
    """Returns the training loss of final model after training."""
    return self._final_loss

  def _get_trainable_x(self, len_data: int, t0: int,
                       flatten_future: bool = False) -> np.ndarray:
    """Reformulates the sequence of predictors that fits tensorflow model.

    Args:
      len_data: specifies the length of a time window starting from the first
        confirmed case.
      t0: specifies the number of days between the occurrence of the first
        infected case (patient zero) and the first observed case.
      flatten_future: this parameter takes effect in prediction only,
        indicating whether the infection rate or death rate is flattened in
        the future.

    Raises:
      ValueError: if the value of "len_data" is less than the length of
        training data.

    Returns:
      A 2d array in which the first shape equals the value of "len_data" and
      the second shape is the number of pieces (in piecewise model) + 1. Its
      role can be simply considered as "x" in a keras "fit()" method.

    """
    len_training_set = sum(self._knots)
    if len_data < len_training_set:
      raise ValueError("The 'len_data' argument is expected to be no less "
                       f"than the length of training set {len_training_set}.")
    pad_num = np.cumsum(1 - np.array(self._knots_connect))
    output = [[np.pad([1.], [0, len(self._knots) + pad_num[-1]])] * t0]
    for i, l in enumerate(self._knots):
      cont = self._knots_connect[i]  # 1 is continuous and 0 is discontinuous.
      output.append(
          [np.pad(
              [1 - (t + cont) / (l - 1 + cont), (t + cont) / (l - 1 + cont)],
              [i + pad_num[i],
               len(self._knots) - 1 + pad_num[-1] - i - pad_num[i]]
          ) for t in range(l)])
    if len_data > len_training_set:
      cont = self._knots_connect[-1]
      if flatten_future:
        output.append(
            [np.pad(
                [1], [len(self._knots) + pad_num[-1], 0]
            ) for t in range(len_data - len_training_set)])
      else:
        output.append(
            [np.pad(
                [- (t + 1) / (l - 1 + cont), 1 + (t + 1) / (l - 1 + cont)],
                [len(self._knots) - 1 + pad_num[-1], 0]
            ) for t in range(len_data - len_training_set)])
    return np.concatenate(output)

  @staticmethod
  def _setup_callbacks(
      message: Text, t0: int, enable_tensorboard: bool = True,
      tensorboard_logdir: Optional[Text] = None) -> Tuple[
          List[Callback], Text]:
    """Setups the callbacks to monitor model estimation.

    Args:
      t0: specifies the number of days between the occurrence of the first
        infected case (patient zero) and the first observed case.
      message: optionally pass a prefix string in the filenames of training
        weights (in the format of hdf5 file). We will generate a lot of such
        files in the training process.
      enable_tensorboard: whether or not use tensorboard to monitor training.
      tensorboard_logdir: xxx.

    Returns:
      callbacks : TYPE DESCRIPTION.
      min_loss_filepath: TYPE DESCRIPTION.

    """
    min_loss_filepath = os.path.join(
        tempfile.tempdir, message + str(t0) + "min_loss.hdf5")
    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            min_loss_filepath, monitor="loss", save_best_only=True,
            save_weights_only=False))
    callbacks.append(TerminateOnNaN())
    if enable_tensorboard:
      if not os.path.isdir(tensorboard_logdir):
        os.makedirs(tensorboard_logdir)
      logdir = os.path.join(
          tensorboard_logdir,
          f"job{t0}_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
      callbacks.append(TensorBoard(log_dir=logdir))
    return callbacks, min_loss_filepath

  def _fit_with_t0(
      self, data: Sequence[int], t0: int, message: Text,
      enable_tensorboard: bool = False,
      tensorboard_logdir: Optional[Text] = None) -> Tuple[
          Covid19InfectionsPredictModel, TensorType]:
    """Returns the model after training with a given t0.

    Args:
      data: training data (number of daily new confirmed cases) in a 1d array.
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
    model = Covid19InfectionsPredictModel(
        n_weights=2 * len(self._knots) + 1 - sum(self._knots_connect),
        t0=t0,
        len_inputs=len(data) + t0,
        max_latency=self._estimator_args.get(
            "max_latency", DEFAULT_MAX_LATENCY),
        **self._model_args)

    x = self._get_trainable_x(len(data), t0)
    # Pad t0 elements at front to be 0.
    y = np.pad(data, [t0, 0]).astype(np.float64)

    # Define the loss function for each t0 value. Compare the square-root
    # difference.
    def custom_loss(y_actual, y_pred):
      return self._estimator_args.get(
          "loss_function", tf.keras.losses.MSE)(
              tf.math.sqrt(y_actual[t0:]), tf.math.sqrt(y_pred[t0:]))
    optimizer_option = self._estimator_args.get(
        "optimizer", tf.keras.optimizers.Adam)
    optimizer = optimizer_option(
        learning_rate=self._estimator_args.get("learning_rate", 0.01),
        clipnorm=1.0)
    model.compile(optimizer, custom_loss)

    callbacks, min_loss_filepath = Covid19InfectionsEstimator._setup_callbacks(
        message, t0, enable_tensorboard, tensorboard_logdir)
    model.fit(
        x, y, epochs=self._estimator_args.get("epochs", 100),
        batch_size=len(data) + t0, shuffle=False,
        verbose=self._estimator_args.get("verbose", 0),
        callbacks=callbacks)
    model.load_weights(min_loss_filepath)
    loss = custom_loss(y, model(x))
    return model, loss

  def fit(self, data: Sequence[int], min_t0: int = 1,
          max_t0: int = DEFAULT_MAX_LATENCY,
          message: Text = "", enable_tensorboard: bool = False,
          tensorboard_logdir: Optional[Text] = None):
    """Fits data with a list of possible t0 and returns the trained models.

    Args:
      data: training data (number of daily new confirmed cases) in a 1d array.
      message: optionally pass a prefix string in the filenames of training
        weights (in the format of hdf5 file). We will generate a lot of such
        files in the training process.
      enable_tensorboard: whether or not use tensorboard to monitor training.
      tensorboard_logdir: xxx.

    Raises:
      ValueError: if the length of training data mismatch with the "knots"
        argument.

    """
    if sum(self._knots) != len(data):
      raise ValueError(
          "The elements in vector 'knots' should sum up to the the length of "
          f"data {len(data)}, but got value {sum(self._knots)}.")
    results = [
        self._fit_with_t0(
            data, t0, message, enable_tensorboard,
            tensorboard_logdir) for t0 in range(
                min_t0, max_t0 + 1)]
    trained_models, trained_loss = zip(*(results))
    trained_loss = np.array(trained_loss)
    trained_loss[np.isnan(trained_loss)] = np.Inf
    self._final_loss = min(trained_loss)
    self._final_model = trained_models[trained_loss.argmin()]

  def predict(self, duration: int, observed_only: bool = True,
              flatten_future: bool = False
              ) -> Optional[TensorType]:
    """Predicts the number of daily new infected cases or observed cases only.

    Args:
      duration: specifies the number of days for prediction.
      observed_only: if True we calculates the number of observed / confirmed
        cases on each day. Otherwise, it gives the number of new infected cases
        which include those asymptomatic individuals.
      flatten_future: this parameter takes effect in prediction only,
        indicating whether the infection rate or death rate is flattened in
        the future.

    Returns:
      The number of daily new observed cases if "observed_only" is True, or
      the number of daily new infected individuals if "observed_only" is False.

    """
    if self._final_model is None:
      return None
    x_pred = self._get_trainable_x(
        duration, self._final_model.t0, flatten_future)
    if observed_only:
      return self._final_model.daily_observed(x_pred)[self._final_model.t0:]
    else:
      return self._final_model.daily_infected(x_pred)[self._final_model.t0:]

  def get_infect_rate_features(self, duration: int,
                               flatten_future: bool = False
                               ) -> Optional[Tuple[TensorType, TensorType]]:
    """Gets both time-varying infection rate and basic reproduction number.

    Args:
      duration: specifies the number of days for prediction.
      flatten_future: this parameter takes effect in prediction only,
        indicating whether the infection rate or death rate is flattened in
        the future.

    Returns:
      Both the array of infection rates and the array of corresponding basic
      reproduction number.

    """
    if self._final_model is None:
      return None
    x_pred = self._get_trainable_x(
        duration, self._final_model.t0, flatten_future)
    return self._final_model.infection_rate(x_pred)[
        self._final_model.t0:], self._final_model.reproduction_number(x_pred)[
            self._final_model.t0:]

"""Python module for COVID-19 Survival-Convolution Model."""
import numpy as np
import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Text, Tuple
import sklearn
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
tf.keras.backend.set_floatx('float64')

_DEFAULT_LATENCY = 21
_AVE_DAYS_BEFORE_DIAGNOSE = 5.2


class Covid19CasesPredictModel(tf.keras.Model):
  """A unified model to predicts the number of COVID-19 positive cases.

  Attributes:
    h: hazard rate parameter used in defining the survival function that
      specifies the average of time until an infected patient becoming
      symptomatic and being diagnosed positive.

   a: vector that indicates the infection rate at each knot. The length equals
     the length of 'knots' + 1.

   t0: number of days between the occurrence of the first infected case (the
     patient-zero) and the first observed case.

  """

  def __init__(self, n_pieces: int, t0: int, len_inputs: int,
               latency: Optional[int] = _DEFAULT_LATENCY, **kwargs):
    """Initializes an instance of Covid19CasesPredictModel object.

    Args:
      n_pieces: specifies the number of segments in piecewise linear infection
        rate, excluding the constant piece before the first observed case. The
        value is same as the length of attribute "knots" in class
        Covid19CasesEstimator.

      t0: specifies the number of days between the occurrence of the first
        infected case (patient zero) and the first observed case.

      len_inputs: equals the sum of t0 and length of the observed training
        data.

      latency: the maximum number of days for any infected patient to be
        asymptomatic.

      **kwargs: pass additional arguments to the model.

    """
    self._n_pieces = n_pieces
    self._t0 = t0
    self._len_inputs = len_inputs
    self._latency = latency
    self._daily_infected_initializer = tf.constant(
        np.append(np.zeros(self._latency - 1), 1), dtype=tf.float64)
    self.h = tf.Variable(
        kwargs.pop("initial_guess_h", 1. / _AVE_DAYS_BEFORE_DIAGNOSE),
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
            0.2 * tf.ones([self._n_pieces + 1], dtype=tf.float64)),
        name="a")
    super(Covid19CasesPredictModel, self).__init__(**kwargs)

  @property
  def t0(self) -> int:
    """Estimates time between the patient zero and the first observed case."""
    return self._t0

  def get_survival_rate(self) -> tf.Tensor:
    """Returns an array of survival rates based on exponential distribution."""
    surv = tf.math.exp(
        tf.multiply(
            -self.h,
            tf.constant(list(range(self._latency)), dtype=tf.float64)))
    surv = (surv - surv[self._latency - 1]) / (1. - surv[self._latency - 1])
    return surv[::-1]

  def _daily_infected(self, inputs: np.ndarray) -> tf.Tensor:
    """Returns the number of daily new infected cases in a matrix.

    Args:
      inputs: a reformulated sequence of predictors that fits tensorflow model.
        For the detailed formats, see method "_get_trainable_x()" in class
        Covid19CasesEstimator.

    Returns:
      A 2d tensor for computing the number of daily new infections in an
      iterative manner. The first shape is same as the length of argument
      "inputs" and the second shape equals the latency period defined in model.

    """
    surv = self.get_survival_rate()

    # Define a helper function that generates the model-based value of new
    # infections on today based on past history of daily new infections. In
    # this model we adopt a survival-rate convolution model.
    def _evolve_daily_infected(input_a, input_b):
      past_daily_infected = tf.slice(
          input_a, begin=[1], size=[self._latency - 1])
      return tf.concat(
          [past_daily_infected,
           tf.reshape(
               tf.nn.relu(tf.tensordot(
                   self.a, input_b, 1)) * tf.tensordot(
                       past_daily_infected, surv[:-1], 1), [1])], 0)
    return tf.concat([
        tf.reshape(self._daily_infected_initializer, [1, self._latency]),
        tf.scan(fn=_evolve_daily_infected, elems=inputs,
                initializer=self._daily_infected_initializer)[:-1]], 0)

  def daily_infected(self, inputs: np.ndarray) -> tf.Tensor:
    """Returns the number of new infected cases per day in a sequence.

    Args:
      inputs: a reformulated sequence of predictors that fits tensorflow model.
        For the detailed formats, see method "_get_trainable_x()" in class
        Covid19CasesEstimator.

    Returns:
      A 1d tensor for storing the number of daily new infections. The length is
      same as the length of argument "inputs".

    """
    return tf.reshape(
        tf.slice(
            self._daily_infected(inputs),
            begin=[0, self._latency - 1], size=[inputs.shape[0], 1]),
        [inputs.shape[0]])

  def daily_observed(self, inputs: np.ndarray,
                     static_tensorshape: Optional[bool] = False):
    """Returns the number of daily new observed (confirmed) cases in an array.

    Args:
      inputs: a reformulated sequence of predictors that fits tensorflow model.
        For the detailed formats, see method "_get_trainable_x()" in class
        Covid19CasesEstimator.
      static_tensorshape: if True, this method will return tensor with a fixed
        length in training a model. Otherwise, the length can be any positive
        integer, which is used in predicting the number of new confirmed cases
        in future.

    Returns:
      A 1d tensor for storing the number of daily new infections. The length is
      same as the length of argument "inputs".

    """
    inputs_len = self._len_inputs if static_tensorshape else inputs.shape[0]
    surv = self.get_survival_rate()
    return tf.reshape(
        tf.matmul(
            tf.slice(
                self._daily_infected(inputs), begin=[0, 1],
                size=[inputs_len, self._latency - 1]
            ),
            tf.reshape(surv[1:] - surv[:-1], [self._latency - 1, 1])
        ), [inputs_len])

  def infection_rate(self, inputs: np.ndarray) -> tf.Tensor:
    """Estimates time-varying infection rates."""
    return tf.nn.relu(tf.reshape(
        tf.matmul(inputs, tf.reshape(self.a, [self._n_pieces + 1, 1])),
        [inputs.shape[0]]))

  def reproduction_number(self, inputs: np.ndarray) -> tf.Tensor:
    """Estimates time-varying basic reproduction numbers."""
    return tf.squeeze(
        tf.nn.conv1d(
            tf.reshape(self.infection_rate(inputs), [1, inputs.shape[0], 1]),
            tf.reshape(self.get_survival_rate()[::-1], [self._latency, 1, 1]),
            1, "VALID"))

  def call(self, inputs: np.ndarray) -> tf.Tensor:
    """Returns the number of new confirmed cases per day.

    This method needs to be overriden to subclass keras.Model.

    Args:
      inputs: a reformulated sequence of predictors that fits tensorflow model.
        For the detailed formats, see method "_get_trainable_x()" in class
        Covid19CasesEstimator.

    Returns:
      A 1d tensor for storing the number of daily new confirmed cases. The
      length is same as the length of argument "inputs".

    """
    return self.daily_observed(inputs, True)


class Covid19CasesEstimator(sklearn.base.BaseEstimator):
  """Selects the best epidemic model based on observable new cases.

  Attributes:
    final_model: the best Covid19CasesPredictModel after training on the
      history of daily new confirmed cases.
    final_loss: the value of loss function given the final model we obtain
      after training.

  """

  def __init__(self, knots: List[int], estimator_args: Dict[Text, Any],
               **kwargs):
    """Initializes a Covid19CaseEstimator instance.

    Args:
      knots: a list of integers in which each represents the length of one
        piece in the piecewise linear infection rate model. These integers
        should sum up to the length of training data.
      estimator_args: change to non-default arguments for training this
        estimator. User may change: the choice of optimizer, learning rate,
        loss function, and the number of training epochs.
      **kwargs: pass additional arguments to the estimator. For example, the
        initial values in a Covid19CasesPredictModel.

    Raises:
      ValueError: if any number in "knots" is zero or negative.

    Returns:
      An instance of Covid19CasesEstimator class.

    """
    self._knots = knots
    if min(self._knots) <= 0:
      raise ValueError("The elements in vector 'knots' must be positive.")
    self._estimator_args = estimator_args
    self._model_args = kwargs
    self._final_model = None
    self._final_loss = None
    super(Covid19CasesEstimator, self).__init__()

  @property
  def final_model(self) -> Optional[Covid19CasesPredictModel]:
    """Returns the final model after training with different t0."""
    return self._final_model

  @property
  def final_loss(self) -> Optional[tf.Tensor]:
    """Returns the training loss of final model after training."""
    return self._final_loss

  def _get_trainable_x(self, len_data: int, t0: int) -> np.ndarray:
    """Reformulates the sequence of predictors that fits tensorflow model.

    Args:
      len_data: specifies the length of a time window starting from the first
        confirmed case.
      t0: specifies the number of days between the occurrence of the first
        infected case (patient zero) and the first observed case.

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
    output = [[np.pad([1.], [0, len(self._knots)])] * t0]
    for i, l in enumerate(self._knots):
      output.append(
          [np.pad(
              [1. - (t + 1) / l, (t + 1) / l], [i, len(self._knots) - i - 1]
          ) for t in range(l)]
      )
    if len_data > len_training_set:
      output.append(
          [np.pad(
              [- (t + 1) / self._knots[-1], 1. + (t + 1) / self._knots[-1]],
              [len(self._knots) - 1, 0]
          ) for t in range(len_data - len_training_set)]
      )
    return np.concatenate(output)

  def _fit_with_t0(
      self, data: Sequence[int], t0: int, message: Text) -> Tuple[
          Covid19CasesPredictModel, tf.Tensor]:
    """Returns the model after training with a given t0.

    Args:
      data: training data (number of daily new confirmed cases) in a 1d array.
      t0: specifies the number of days between the occurrence of the first
        infected case (patient zero) and the first observed case.
      message: optionally pass a prefix string in the filenames of training
        weights (in the format of hdf5 file). We will generate a lot of such
        files in the training process.

    Returns:
      model: the best model after training with t0.
      loss: the loss of the best model after training with t0.

    """
    model = Covid19CasesPredictModel(
        n_pieces=len(self._knots), t0=t0, len_inputs=len(data) + t0,
        latency=self._estimator_args.get("latency", _DEFAULT_LATENCY),
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
        learning_rate=self._estimator_args.get("learning_rate", 0.01))
    model.compile(optimizer, custom_loss)
    min_loss_filepath = os.path.join(
        tempfile.tempdir, message + str(t0) + "min_loss.hdf5")
    mcp = ModelCheckpoint(min_loss_filepath, monitor="loss",
                          save_best_only=True, save_weights_only=False)
    model.fit(
        x, y, epochs=self._estimator_args.get("epochs", 100),
        batch_size=len(data) + t0, shuffle=False, verbose=0, callbacks=[mcp])
    model.load_weights(min_loss_filepath)
    loss = custom_loss(y, model(x))
    return model, loss

  def fit(self, data: Sequence[int], min_t0: Optional[int] = 1,
          max_t0: Optional[int] = _DEFAULT_LATENCY,
          message: Optional[Text] = ""):
    """Fits data with a list of possible t0 and returns the trained models.

    Args:
      data: training data (number of daily new confirmed cases) in a 1d array.
      message: optionally pass a prefix string in the filenames of training
        weights (in the format of hdf5 file). We will generate a lot of such
        files in the training process.

    Raises:
      ValueError: if the length of training data mismatch with the "knots"
        argument.

    """
    if sum(self._knots) != len(data):
      raise ValueError(
          "The elements in vector 'knots' should sum up to the the length of"
          f"data {len(data)}, but got value {sum(self._knots)}.")
    results = [
        self._fit_with_t0(data, t0, message) for t0 in range(
            min_t0, max_t0 + 1)]
    trained_models, trained_loss = zip(*(results))
    self._final_loss = min(trained_loss)
    self._final_model = trained_models[trained_loss.index(self._final_loss)]

  def predict(self, duration: int,
              observed_only: Optional[bool] = True) -> Optional[tf.Tensor]:
    """Predicts the number of daily new infected cases or observed cases only.

    Args:
      duration: specifies the number of days for prediction.
      observed_only: if True we calculates the number of observed / confirmed
        cases on each day. Otherwise, it gives the number of new infected cases
        which include those asymptomatic individuals.

    Returns:
      The number of daily new observed cases if "observed_only" is True, or
      the number of daily new infected individuals if "observed_only" is False.

    """
    if self._final_model is None:
      return None
    x_pred = self._get_trainable_x(duration, self._final_model.t0)
    if observed_only:
      return self._final_model.daily_observed(x_pred)[self._final_model.t0:]
    else:
      return self._final_model.daily_infected(x_pred)[self._final_model.t0:]

  def get_infect_rate_features(self, duration: int
                               ) -> Optional[Tuple[tf.Tensor, tf.Tensor]]:
    """Gets both time-varying infection rate and basic reproduction number.

    Args:
      duration: specifies the number of days for prediction.

    Returns:
      Both the array of infection rates and the array of corresponding basic
      reproduction number.

    """
    if self._final_model is None:
      return None
    x_pred = self._get_trainable_x(duration, self._final_model.t0)
    return self._final_model.infection_rate(x_pred)[
        self._final_model.t0:], self._final_model.reproduction_number(x_pred)[
            self._final_model.t0:]

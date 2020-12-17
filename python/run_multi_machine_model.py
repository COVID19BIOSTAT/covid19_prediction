"""Python binary to train Covid-19 cases model with multiple CPUs."""

from absl import app
from absl import flags
import functools
import multiprocessing
import numpy as np
import os
import piecewise_linear_infection_model
import tensorflow as tf
from typing import List, Tuple

flags.DEFINE_list("initial_a", None, "Initial values of vector a.")
flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate.")
flags.DEFINE_integer("max_epochs", 2000, "Number of steps to run trainer.")
flags.DEFINE_integer(
    "n_process", 10, "Number of parallel jobs to run trainer.")
flags.DEFINE_integer("test_duration", 100, "Number of days to predict")
flags.DEFINE_list("knots", None, "Knots in piecewise model.")
flags.DEFINE_string("input_path", None, "Path to store the input data.")
flags.DEFINE_string("output_path", None, "Path to store the output data.")
flags.DEFINE_integer("min_t0", 1, "value for which t0 iteration starts.")
flags.DEFINE_integer("max_t0", 21, "value for which t0 iteration ends.")
FLAGS = flags.FLAGS


def train_single_model(
        t0: int,
        data: np.ndarray,
        knots: List[int],
        learning_rate: np.float64,
        max_epochs: int,
        initial_a: List[np.float64]) -> Tuple[
            piecewise_linear_infection_model.Covid19CasesPredictModel,
            tf.Tensor]:
  """Trains single Covid19CasesEstimator with one worker.

  Args:
    t0: specifies the number of days between the occurrence of the first
      infected case (patient zero) and the first observed case.
    data: training data (number of daily new confirmed cases) in a 1d array.
    knots : a list of integers in which each represents the length of one
      piece in the piecewise linear infection rate model. These integers should
      sum up to the length of training data.
    learning_rate : the learning rate in an optimizer.
    max_epochs : the number of steps to run trainer.
    initial_a : a list of float numbers indicating the infection rate at each
      knot. The length equals the length of 'knots' + 1.

  Returns:
    model: the best model after training with t0.
    loss: the loss of the best model after training with t0.

  """
  estimator = piecewise_linear_infection_model.Covid19CasesEstimator(
      knots=knots,
      estimator_args={
          "learning_rate": learning_rate,
          "epochs": max_epochs},
      initial_guess_a=np.array(initial_a)
  )
  model, loss = estimator._fit_with_t0(data, t0, "")
  return model.get_weights(), loss


def main(unused_argv):
  input_data = np.load(FLAGS.input_path)
  pool_workers = multiprocessing.Pool(FLAGS.n_process)
  parsed_knots = [int(k) for k in FLAGS.knots]
  parsed_initial_a = [np.float64(a) for a in FLAGS.initial_a]

  target_fn = functools.partial(
      train_single_model,
      data=input_data, knots=parsed_knots,
      learning_rate=FLAGS.learning_rate,
      max_epochs=FLAGS.max_epochs,
      initial_a=parsed_initial_a)

  all_t0 = list(range(FLAGS.min_t0, FLAGS.max_t0 + 1))
  all_results = pool_workers.map(func=target_fn, iterable=all_t0)
  trained_weights, trained_loss = zip(*(all_results))

  best_loss = min(trained_loss)
  best_weights = trained_weights[trained_loss.index(best_loss)]
  best_t0 = all_t0[trained_loss.index(best_loss)]
  print(f"The selected t0 = {best_t0}, the corresponding weights are "
        f"{best_weights} and loss = {best_loss}.")
  estimator = piecewise_linear_infection_model.Covid19CasesEstimator(
      knots=parsed_knots, estimator_args={})
  best_model = piecewise_linear_infection_model.Covid19CasesPredictModel(
      n_pieces=len(parsed_knots), t0=best_t0,
      len_inputs=len(input_data) + best_t0)
  best_model.set_weights(best_weights)
  estimator._final_model = best_model

  if not os.path.exists(FLAGS.output_path):
    os.makedirs(FLAGS.output_path)
    
  np.save(file=os.path.join(FLAGS.output_path, "predicted_daily_observed"),
          arr=estimator.predict(FLAGS.test_duration, True).numpy())
  np.save(file=os.path.join(FLAGS.output_path, "predicted_daily_infected"),
          arr=estimator.predict(FLAGS.test_duration, False).numpy())
  np.save(file=os.path.join(FLAGS.output_path, "best_weights"),
          arr=best_weights)
  np.save(file=os.path.join(FLAGS.output_path, "best_t0"),
          arr=best_t0)
  infection_rate_features = estimator.get_infect_rate_features(
      FLAGS.test_duration)
  np.save(file=os.path.join(FLAGS.output_path, "predicted_infection_rate"),
          arr=infection_rate_features[0].numpy())
  np.save(
      file=os.path.join(FLAGS.output_path, "predicted_reproduction_number"),
      arr=infection_rate_features[1].numpy())


if __name__ == "__main__":
  flags.mark_flag_as_required("knots")
  flags.mark_flag_as_required("input_path")
  flags.mark_flag_as_required("output_path")
  flags.mark_flag_as_required("initial_a")
  app.run(main)

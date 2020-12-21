r"""Python binary to train Covid-19 cases model with multiple CPUs.

Example of running the model:
python run_multi_machine_model.py --knots=22,14,14,21,21,35,28 \
  --input_path="real_data.npy" --output_path="output" --n_process=4 \
  --max_epochs=300 --initial_a=0.416,0.829,0.325,0.216,0.222,0.213,0.282,0.2 \
  --loss_prop=0.9 --learning_rate=0.01 --test_duration=250 --min_t0=14 \
  --max_t0=16
"""
from absl import app
from absl import flags
from typing import List, Text, Tuple
import io_utils
import functools
import logging
import multiprocessing
import numpy as np
import os
# pylint: disable=import-not-at-top
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import piecewise_linear_infection_model as infection_model
import combined_model

flags.DEFINE_integer("max_epochs", 2000, "Number of steps to run trainer.")
flags.DEFINE_integer(
    "n_process", 10, "Number of parallel jobs to run trainer.")
flags.DEFINE_integer("min_t0", 1, "Value for which t0 iteration starts.")
flags.DEFINE_integer("max_t0", 21, "Value for which t0 iteration ends.")
flags.DEFINE_integer("test_duration", 100, "Number of days to predict")
flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate.")
flags.DEFINE_float("loss_prop", 0.9, "Loss weight assigned to death model.")
flags.DEFINE_list("initial_a", None, "Initial values of vector a.")
flags.DEFINE_list("knots", None, "Knots in piecewise model.")
flags.DEFINE_list(
    "knots_connect", None, "Whether a knot is connected with the last piece.")
flags.DEFINE_string("input_path", None, "Path to store the input data.")
flags.DEFINE_string("output_path", None, "Path to store the output data.")
flags.DEFINE_bool("flatten_future", True, "Indicates whether or not the "
                  "future prediction relies on the flat death and infection "
                  "rates.")
flags.DEFINE_bool(
    "output_to_json", True, "If the export data are saved in json file.")
flags.DEFINE_bool(
    "enable_tensorboard", True, "If tensorboard is enabled to monitor model "
    "estimation.")
flags.DEFINE_string(
    "tensorboard_logdir", "logs", "The directory that saves tensorboard "
    "training logs.")

FLAGS = flags.FLAGS


def train_single_model(
        t0: int,
        data: np.ndarray,
        knots: List[int],
        knots_connect: List[int],
        learning_rate: np.float64,
        loss_prop: np.float64,
        max_epochs: int,
        initial_a: List[np.float64],
        enable_tensorboard: bool,
        tensorboard_logdir: Text) -> Tuple[
            combined_model.Covid19CombinedPredictModel,
            infection_model.TensorType]:
  """Trains single Covid19CombinedEstimator with one worker.

  Args:
    t0: specifies the number of days between the occurrence of the first
      infected case (patient zero) and the first observed case.
    data: training data (number of daily new confirmed cases) in a 1d array.
    knots: a list of integers in which each represents the length of one piece
      in the piecewise linear infection rate model. These integers should sum
      up to the length of training data.
    knots_connect:.
    learning_rate : the learning rate in an optimizer.
    loss_prop:.
    max_epochs : the number of steps to run trainer.
    initial_a : a list of float numbers indicating the infection rate at each
      knot. The length equals the length of 'knots' + 1.

  Returns:
    weights: the weights in the best model after training with t0.
    loss: the loss of the best model after training with t0.

  """
  estimator = combined_model.Covid19CombinedEstimator(
      knots=knots,
      knots_connect=knots_connect,
      loss_prop=loss_prop,
      estimator_args={
          "learning_rate": learning_rate,
          "epochs": max_epochs},
      initial_guess_a=np.array(initial_a)
  )
  model, loss = estimator._fit_with_t0(
      data, t0, "", enable_tensorboard, tensorboard_logdir)
  return model.weights, loss


def main(unused_argv):
  """Runs model estimation in parallel."""
  logger = logging.getLogger("covid-19-estimation")
  input_data = np.load(FLAGS.input_path)
  if FLAGS.loss_prop == 0:
    logger.warning(
        "The death data are not used for model training.")
  if FLAGS.loss_prop == 1:
    logger.warning(
        "The daily observed cases are not used for model training.")
  logger.info("Loaded the training data.")
  parsed_knots, parsed_knots_connect, parsed_initial_a = io_utils.parse_knots(
      FLAGS.knots, FLAGS.knots_connect, FLAGS.initial_a)
  pool_workers = multiprocessing.Pool(FLAGS.n_process)
  if sum(parsed_knots) != input_data.shape[0]:
    raise ValueError(
        "The elements in vector 'knots' should sum up to the the length of "
        f"data {input_data.shape[0]}, but got value {sum(parsed_knots)}.")

  target_fn = functools.partial(
      train_single_model,
      data=input_data,
      knots=parsed_knots,
      knots_connect=parsed_knots_connect,
      learning_rate=FLAGS.learning_rate,
      loss_prop=FLAGS.loss_prop,
      max_epochs=FLAGS.max_epochs,
      initial_a=parsed_initial_a,
      enable_tensorboard=FLAGS.enable_tensorboard,
      tensorboard_logdir=FLAGS.tensorboard_logdir)

  all_t0 = list(range(FLAGS.min_t0, FLAGS.max_t0 + 1))
  all_results = pool_workers.map(func=target_fn, iterable=all_t0)
  logger.info("Completed model estimation.")

  trained_weights, trained_losses = zip(*(all_results))
  remove_nan_losses = np.array(trained_losses)
  remove_nan_losses[np.isnan(remove_nan_losses)] = np.Inf
  best_weights = trained_weights[remove_nan_losses.argmin()]

  estimator_template = combined_model.Covid19CombinedEstimator(
      knots=parsed_knots,
      knots_connect=parsed_knots_connect,
      estimator_args={})
  selected_t0 = all_t0[remove_nan_losses.argmin()]
  model_template = combined_model.Covid19CombinedPredictModel(
      n_weights=len(parsed_initial_a),
      t0=selected_t0,
      len_inputs=len(input_data) + selected_t0)
  model_template.load_pretrained_weights(best_weights)
  estimator_template._final_model = model_template
  estimator_template._final_loss = min(remove_nan_losses)

  io_utils.parse_estimated_model(estimator_template)
  io_utils.export_estimation_and_prediction(
      estimator=estimator_template,
      test_duration=FLAGS.test_duration,
      output_path=FLAGS.output_path,
      flatten_future=FLAGS.flatten_future,
      to_json=FLAGS.output_to_json
  )
  logger.info("Done.")


if __name__ == "__main__":
  flags.mark_flag_as_required("knots")
  flags.mark_flag_as_required("input_path")
  flags.mark_flag_as_required("output_path")
  app.run(main)

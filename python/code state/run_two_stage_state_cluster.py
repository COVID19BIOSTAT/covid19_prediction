#!/user/bin/env python
"""Python binary to train COVID-19 epidemic model with single CPU."""
from typing import Text
from absl import app
from absl import flags
import io_utils
import json
import logging
import numpy as np
import os
import resample
# pylint: disable=import-not-at-top
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import combined_model

NUM_STATES = 58
flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate.")
flags.DEFINE_integer("max_epochs", 2000, "Number of steps to run trainer.")
flags.DEFINE_integer("test_duration", 100, "Number of days to predict")
flags.DEFINE_string("metadata", None, "")
flags.DEFINE_string("residuals", None, "Only used in resample jobs.")
flags.DEFINE_string("output_path", None, "Path to store the output data.")
flags.DEFINE_integer("min_t0", 1, "value for which t0 iteration starts.")
flags.DEFINE_integer("max_t0", 21, "value for which t0 iteration ends.")
flags.DEFINE_string("file_index_plus_one", None, "Index for permutation.")
flags.DEFINE_bool(
    "output_to_json", True, "If the export data are saved in json file.")
flags.DEFINE_bool(
    "resample_jobs", False, "If the model estimation is on resampled data.")
flags.DEFINE_bool("flatten_future", True, "Indicates whether or not the "
                  "future prediction relies on the flat death and infection "
                  "rates.")
FLAGS = flags.FLAGS


def read_metadata(fpath: Text, n_states: int):
  with open(fpath, "r") as f:
    jdata = json.load(f)
  if len(jdata.keys()) != n_states:
    raise ValueError(f"The metadata should contains exactly {n_states} records.")
  for k, v in jdata.items():
    if set(v.keys()) != {"state", "knot", "data"}:
      raise ValueError(f"At the record {k}, the json data don't follow the required format, "
                       "expected to have three fields: state, knot, and data.")
  return jdata


def map_keys(file_index: int, n_states: int, resample_jobs: bool = False):
  int_file_index = int(file_index)
  if resample_jobs:
    return str(int_file_index % n_states), str(int_file_index // n_states)
  else:
    if int_file_index >= n_states:
      raise ValueError(f"The file index should be between 0 and {n_states}.")
    return file_index, None


def main(unused_argv):
  logger = logging.getLogger("Covid-19-estimation")
  job_metadata = read_metadata(FLAGS.metadata, NUM_STATES)
  job_file_index = str(int(FLAGS.file_index_plus_one) - 1)
  state_key, resample_key = map_keys(
      job_file_index, NUM_STATES, FLAGS.resample_jobs)
  job_state = job_metadata[state_key]["state"]
  logger.info(f"Create estimation job for the state {job_state}.")

  # Parse the required data & args for running the estimation model.
  job_knots = job_metadata[state_key]["knot"]
  if FLAGS.resample_jobs:
    job_residuals = resample.read_residuals(
        FLAGS.residuals, NUM_STATES)
    job_input_data = resample.get_resampled_input(
        job_residuals[state_key]["fitted"],
        job_residuals[state_key]["residual"],
        int(resample_key))
  else:
    job_input_data = job_metadata[state_key]["data"]
  job_knots_connect = [1] * len(job_knots)
  job_initial_a = [0.2] * (len(job_knots) + 1)

  # Step 1 estimation (infection cases only).
  estimator = combined_model.Covid19CombinedEstimator(
      knots=job_knots,
      knots_connect=job_knots_connect,
      loss_prop=0,
      estimator_args={
          "learning_rate": FLAGS.learning_rate,
          "epochs": FLAGS.max_epochs},
      initial_guess_a=np.array(job_initial_a),
      variable_death_rate_trainable=False
  )
  estimator.fit(data=job_input_data, min_t0=FLAGS.min_t0, max_t0=FLAGS.max_t0)
  # Save the estimated weights from step 1 (will be fixed in step 2).
  stage1_estimated_a, stage1_estimated_t0 = estimator.final_model.a.numpy(), estimator.final_model.t0
  logger.info("First stage estimation done.")

  # Step 2 estimation (death only).
  estimator = combined_model.Covid19CombinedEstimator(
      knots=job_knots,
      knots_connect=job_knots_connect,
      loss_prop=1,
      estimator_args={
          "learning_rate": FLAGS.learning_rate,
          "epochs": FLAGS.max_epochs},
      initial_guess_a=stage1_estimated_a,
      variable_a_trainable=False
  )
  estimator.fit(data=job_input_data, min_t0=stage1_estimated_t0, max_t0=stage1_estimated_t0)
  io_utils.parse_estimated_model(estimator)
  if FLAGS.resample_jobs:
    job_suffix = "state_" + state_key + "resample_" + resample_key
  else:
    job_suffix = "state_" + state_key
  io_utils.export_estimation_and_prediction(
      estimator=estimator,
      test_duration=FLAGS.test_duration,
      output_path=FLAGS.output_path,
      suffix=job_suffix,
      flatten_future=FLAGS.flatten_future,
      to_json=FLAGS.output_to_json
  )
  logger.info("Second stage estimation done.")


if __name__ == "__main__":
  flags.mark_flag_as_required("metadata")
  flags.mark_flag_as_required("output_path")
  flags.mark_flag_as_required("file_index_plus_one")
  app.run(main)

#!/user/bin/env python
"""Python binary to train COVID-19 epidemic model with single CPU."""
from absl import app
from absl import flags
import io_utils
import logging
import numpy as np
import os
# pylint: disable=import-not-at-top
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import combined_model


flags.DEFINE_list("initial_a", None, "Initial values of vector a.")
flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate.")
flags.DEFINE_integer("max_epochs", 2000, "Number of steps to run trainer.")
flags.DEFINE_integer("test_duration", 100, "Number of days to predict")
flags.DEFINE_list("knots", None, "Knots in piecewise model.")
flags.DEFINE_list(
    "knots_connect", None, "Whether a knot is connected with the last piece.")
flags.DEFINE_string("input_path", None, "Path to store the input data.")
flags.DEFINE_string("output_path", None, "Path to store the output data.")
flags.DEFINE_integer("min_t0", 1, "value for which t0 iteration starts.")
flags.DEFINE_integer("max_t0", 21, "value for which t0 iteration ends.")
flags.DEFINE_string("file_index", None, "Index for permutation.")
flags.DEFINE_bool("flatten_future", True, "Indicates whether or not the "
                  "future prediction relies on the flat death and infection "
                  "rates.")

FLAGS = flags.FLAGS


def main(unused_argv):
  logger = logging.getLogger("Covid-19-estimation")
  input_data = np.load(FLAGS.input_path)
  parsed_knots, parsed_knots_connect, parsed_initial_a = io_utils.parse_knots(
      FLAGS.knots, FLAGS.knots_connect, FLAGS.initial_a)

  # Step 1 estimation (infection cases only).
  estimator = combined_model.Covid19CombinedEstimator(
      knots=parsed_knots,
      knots_connect=parsed_knots_connect,
      loss_prop=0,
      estimator_args={
          "learning_rate": FLAGS.learning_rate,
          "epochs": FLAGS.max_epochs},
      initial_guess_a=np.array(parsed_initial_a),
      variable_death_rate_trainable=False
  )
  estimator.fit(data=input_data, min_t0=FLAGS.min_t0, max_t0=FLAGS.max_t0)
  # Save the estimated weights from step 1 (will be fixed in step 2).
  stage1_estimated_a, stage1_estimated_t0 = estimator.final_model.a.numpy(), estimator.final_model.t0
  logger.info("First stage estimation done.")

  # Step 2 estimation (death only).
  estimator = combined_model.Covid19CombinedEstimator(
      knots=parsed_knots,
      knots_connect=parsed_knots_connect,
      loss_prop=1,
      estimator_args={
          "learning_rate": FLAGS.learning_rate,
          "epochs": FLAGS.max_epochs},
      initial_guess_a=stage1_estimated_a,
      variable_a_trainable=False
  )
  estimator.fit(data=input_data, min_t0=stage1_estimated_t0, max_t0=stage1_estimated_t0)
  io_utils.parse_estimated_model(estimator)
  io_utils.export_estimation_and_prediction(
      estimator=estimator,
      test_duration=FLAGS.test_duration,
      output_path=FLAGS.output_path,
      suffix=FLAGS.file_index,
      flatten_future=FLAGS.flatten_future,
      to_json=True
  )
  logger.info("Second stage estimation done.")


if __name__ == "__main__":
  flags.mark_flag_as_required("knots")
  flags.mark_flag_as_required("input_path")
  flags.mark_flag_as_required("output_path")
  flags.mark_flag_as_required("file_index")
  app.run(main)

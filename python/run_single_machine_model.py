"""Python binary to train Covid-19 cases model with single CPU."""
from absl import app
from absl import flags
import numpy as np
import os
import piecewise_linear_infection_model

flags.DEFINE_list("initial_a", None, "Initial values of vector a.")
flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate.")
flags.DEFINE_integer("max_epochs", 2000, "Number of steps to run trainer.")
flags.DEFINE_integer("test_duration", 100, "Number of days to predict")
flags.DEFINE_list("knots", None, "Knots in piecewise model.")
flags.DEFINE_string("input_path", None, "Path to store the input data.")
flags.DEFINE_string("output_path", None, "Path to store the output data.")
flags.DEFINE_integer("min_t0", 1, "value for which t0 iteration starts.")
flags.DEFINE_integer("max_t0", 21, "value for which t0 iteration ends.")
FLAGS = flags.FLAGS


def main(unused_argv):
  input_data = np.load(FLAGS.input_path)
  parsed_knots = [int(k) for k in FLAGS.knots]
  parsed_initial_a = [np.float64(a) for a in FLAGS.initial_a]

  estimator = piecewise_linear_infection_model.Covid19CasesEstimator(
      knots=parsed_knots,
      estimator_args={
          "learning_rate": FLAGS.learning_rate,
          "epochs": FLAGS.max_epochs},
      initial_guess_a=np.array(parsed_initial_a)
  )

  estimator.fit(data=input_data, min_t0=FLAGS.min_t0, max_t0=FLAGS.max_t0)

  best_model = estimator.final_model
  best_loss = estimator.final_loss
  best_weights = best_model.get_weights()
  best_t0 = best_model.t0
  print(f"The selected t0 = {best_t0}, the corresponding weights are "
        f"{best_weights} and loss = {best_loss}.")

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

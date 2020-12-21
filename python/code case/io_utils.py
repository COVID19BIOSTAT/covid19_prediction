"""Collection of I/O utility functions for estimating Covid-19 models."""
import json
import os
import re
import numpy as np
import piecewise_linear_infection_model as infection_model
from typing import Any, Dict, List, Optional, Text, Tuple


def parse_knots(
        flags_knots: List[Text],
        flags_knots_connect: List[Text],
        flags_initial_a: List[Text]) -> Tuple[
            List[int], List[int], List[np.float64]]:
  """Parses knots arguments from flags to list of parameters.

  Before parsing, all input data are a list of strings. This function
  transforms them into the corresponding numerical values.

  Args:
    flags_knots: the knots in a piecewise linear infection rate model. Each
      element indicates the length of a piece.
    flags_knots_connect: indicates whether or not the end of a pieces is
      continous.
    flags_initial_a:  vector that indicates the infection rate at each knot.

  Raises:
    ValueError: if the lengths of 'knots' and 'knots_connect' are unequal.

  Returns:
    parsed_knots: the knots in a piecewise linear infection rate model. Each
      element indicates the length of a piece.
    parsed_knots_connect: indicates whether or not the end of a pieces is
      continous.
    parsed_initial_a: vector that indicates the infection rate at each knot.

  """
  parsed_knots = [int(k) for k in flags_knots]

  if flags_knots_connect is None:
    parsed_knots_connect = [1] * len(flags_knots)
  else:
    parsed_knots_connect = [int(k) for k in flags_knots_connect]

  if len(parsed_knots_connect) != len(parsed_knots):
    raise ValueError("The length of knots should be the same as the length"
                     "of knots_connect.")
  n_weights = len(parsed_knots) + 1 + sum(1 - np.array(parsed_knots_connect))
  if flags_initial_a is None:
    parsed_initial_a = [0.5] * n_weights
  else:
    parsed_initial_a = [np.float64(a) for a in flags_initial_a]
  return parsed_knots, parsed_knots_connect, parsed_initial_a


def parse_estimated_model(
        estimator: infection_model.Covid19InfectionsEstimator):
  """Shows the optimal model estimation after training.

  Args:
    estimator: an instance of Covid19InfectionsEstimator, Covid19DeathEstimator
      or Covid19CombinedEstimator.

  """
  print(f"The minimum loss in training = {estimator.final_loss}.")
  print(f"The selected t0 = {estimator.final_model.t0}.")
  estimated_weights = estimator.final_model.weights
  for single_weight in estimated_weights:
    full_name = single_weight.name
    pattern = re.search(r":(\d+)", full_name)
    short_name = full_name[:pattern.start()] if pattern else full_name
    single_weight_arr = single_weight.numpy()
    print(f"The estimated {short_name} = {single_weight_arr.flatten()}.")


class NumpyEncoder(json.JSONEncoder):
  """A helper class to encode numpy arrays to json."""

  def default(self, obj):
    """Required to encode numpy arrays."""
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.floating):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)


def export_estimation_and_prediction(
        estimator: infection_model.Covid19InfectionsEstimator,
        test_duration: int,
        output_path: Text,
        suffix: Optional[Text] = "",
        flatten_future: bool = False,
        to_json: bool = False):
  """Exports estimated infection rates and prediction (infections, deaths).

  Args:
    estimator: an instance of Covid19InfectionsEstimator, Covid19DeathEstimator
      or Covid19CombinedEstimator.
    test_duration: specifies the number of days for prediction. The first day
      in the prediction should be aligned with the time of first observed case
      in training data.
    output_path: specifies the output directory for the predicted values and
      infection rate features.
    suffix: optionally passes a suffix to the output files.
    flatten_future: xxx.
    to_json: if true the export data are saved in a json file; otherwise each
      item in the export is saved in a separate npy file.

  """
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  infection_rate_features = estimator.get_infect_rate_features(
      test_duration, flatten_future)

  export_data = dict(
      predicted_infection_rate=infection_rate_features[0].numpy(),
      predicted_reproduction_number=infection_rate_features[1].numpy(),
      predicted_daily_observed=estimator.predict(
          test_duration, True, flatten_future).numpy(),
      predicted_daily_infected=estimator.predict(
          test_duration, False, flatten_future).numpy(),
      predicted_daily_death=estimator.predict_death(
          test_duration, flatten_future).numpy(),
      best_weights=estimator.final_model.get_weights(),
      best_t0=estimator.final_model.t0
  )

  if to_json:
    dumped = json.dumps(export_data, cls=NumpyEncoder)
    with open(os.path.join(output_path, f"export{suffix}.json"), "w") as f:
      json.dump(dumped, f)
  else:
    for key, value in export_data.items():
      np.save(file=os.path.join(output_path, key + suffix), arr=value)


def parse_json_export(fpath: Text) -> Dict[Text, Any]:
  """Parses export data from a json file into a dict of arrays."""
  with open(fpath, "r") as f:
    jdata = json.load(f)
  return json.loads(jdata)

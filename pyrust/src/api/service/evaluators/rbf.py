from mini_keras import RBFKMeans, RBFNaive
import json

def evaluate_rbf(json_str, input_data):
    config = json.loads(json_str)

    if all(k in config for k in ("x", "y", "gamma")):
        model = RBFNaive.from_json(json_str)
        raw_preds = [model.predict(x) for x in input_data] if isinstance(input_data[0], (list, tuple)) else [model.predict(input_data)]
    elif all(k in config for k in ("centers", "weights", "gamma")):
        model = RBFKMeans.from_json(json_str)
        raw_preds = [model.predict(x) for x in input_data] if isinstance(input_data[0], (list, tuple)) else [model.predict(input_data)]
    else:
        raise ValueError("Non recognized RBF model configuration.")

    threshold = 0.5
    labels = [(-1 if p < threshold else 1) for p in raw_preds]


    return {"prediction": labels}

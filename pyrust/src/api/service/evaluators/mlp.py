from mini_keras import MLP


def evaluate_mlp(json_str: str, input_data: list[float]) -> dict:
    model = MLP.from_json(json_str)
    raw_preds = model.predict(input_data)
    print(f"Raw predictions: {raw_preds}")
    threshold = 0.5
    labels = [(-1 if p < threshold else 1) for p in raw_preds]

    return {"prediction": list(labels)}

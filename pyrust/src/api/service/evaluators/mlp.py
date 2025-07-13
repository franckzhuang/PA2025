from mini_keras import MLP

def evaluate_mlp(json_str: str, input_data: list[float]) -> dict:
    model = MLP.from_json(json_str)
    preds = model.predict(input_data)
    return {"prediction": list(preds)}

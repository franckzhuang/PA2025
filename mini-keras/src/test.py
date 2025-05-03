def mse(num_samples: int, predicted_values: list[int], actual_values: list[int]) -> float:
    return sum([pow(actual - predicted, 2) for actual, predicted in zip(actual_values, predicted_values)]) / num_samples



def predict(features: list[int], weight: list[int], biais: int):
    return sum([f * w for f, w in zip(features, weight)]) + biais


def gradient_descent(x_data: list[list[int]], y_data: list[int], w, b):
    n_samples = len(x_data)
    n_features = len(x_data[0])

    dw = [0.0 for _ in range (n_features)] # init derive
    db = 0.0

    for x, y in zip (x_data, y_data):
        y_pred = predict(x, w, b)

def train(x_data: list[list[int]], y_data: list[int], learning_rate: int = 0.01, epochs: int = 1000):
    n_features = len(x_data[0])
    w = [0.0 for _ in range(n_features)] # init weights
    b = 0.0

    for epoch in range(epochs):
        pass



if __name__ == "__main__":
    print(mse(2, [1, 2], [2, 4]))
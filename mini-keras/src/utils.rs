// #[derive(Debug, Clone, Copy, PartialEq)]
// pub enum ModelMode {
//     Regression,
//     Classification,
// }
//
// impl ModelMode {
//     pub fn from_str(mode_str: &str) -> Result<Self, String> {
//         match mode_str.to_lowercase().as_str() {
//             "regression" => Ok(ModelMode::Regression),
//             "classification" => Ok(ModelMode::Classification),
//             _ => Err(format!(
//                 "Invalid model mode: '{}'. Please, make sure to use 'regression' or 'classification'.",
//                 mode_str
//             )),
//         }
//     }
// }

pub fn check_mode(mode_input: &str) -> Result<String, String> {
    let mode_lowercase = mode_input.to_lowercase();
    if mode_lowercase == "classification" || mode_lowercase == "regression" {
        Ok(mode_lowercase)
    } else {
        Err(format!(
            "Invalid mode: '{}'. Supported modes are 'classification' or 'regression'.",
            mode_input
        ))
    }
}

pub fn mse(predictions: &[f64], actual: &[f64]) -> f64 {
    if predictions.len() != actual.len() {
        panic!(
            "Error: predictions and actual values must have the same length. \
            Pred len: {}, Actual len: {}",
            predictions.len(),
            actual.len()
        );
    }

    if predictions.is_empty() {
        return 0.0;
    }

    let mut sum_squared_error = 0.0;
    for (pred, act) in predictions.iter().zip(actual.iter()) {
        let error = pred - act;
        sum_squared_error += error * error;
    }

    sum_squared_error / predictions.len() as f64
}

pub fn accuracy(predictions: &[f64], actual: &[f64]) -> f64 {
    if predictions.len() != actual.len() {
        panic!(
            "Error: predictions and actual values must have the same length. \
            Pred len: {}, Actual len: {}",
            predictions.len(),
            actual.len()
        );
    }

    if predictions.is_empty() {
        return 0.0;
    }

    let mut correct_count = 0;
    for (pred, act) in predictions.iter().zip(actual.iter()) {
        if (*pred >= 0.0 && *act >= 0.0) || (*pred < 0.0 && *act < 0.0) {
            correct_count += 1;
        }
    }

    correct_count as f64 / predictions.len() as f64
}
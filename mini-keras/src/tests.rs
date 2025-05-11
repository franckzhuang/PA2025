use crate::utils::{accuracy, mse};
use crate::linear_model::LinearModel;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;


    #[test]
    fn test_regression() {
        let x_data = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![4.0, 5.0]
        ];

        let y_data = vec![12.0, 17.0, 22.0, 27.0];

        let mut model = LinearModel::new(0.01, 1000, "regression".to_string(), false);
        model.fit(&x_data, &y_data);

        let y_pred = model.predict(&x_data);

        let mse_value = mse(&y_pred, &y_data);

        println!("Predictions: {:?}", y_pred.iter().map(|p| (p * 100.0).round() / 100.0).collect::<Vec<f64>>());
        println!("MSE: {:.6}", mse_value);

        assert!(mse_value < 1e-2, "MSE is too high: {}", mse_value);
    }

    #[test]
    fn test_classification() {
        let x_data = vec![
            vec![2.0, 1.0],
            vec![-1.0, -1.0],
            vec![1.5, 0.5],
            vec![-2.0, -2.0]
        ];
        let y_data = vec![1.0, -1.0, 1.0, -1.0];

        let mut model = LinearModel::new(0.1, 1000, "classification".to_string(), false);
        model.fit(&x_data, &y_data);

        let y_pred = model.predict(&x_data);

        let acc = accuracy(&y_pred, &y_data);

        println!("Predictions: {:?}", y_pred);
        println!("Accuracy: {:.2}%", acc * 100.0);

        assert_abs_diff_eq!(acc, 1.0, epsilon = 1e-6);
    }
}
use ndarray::{Array1, Array2, s};
// use ndarray_linalg::Inverse;
use serde::{Serialize, Deserialize};
use std::fs::File;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LinearClassification {
    pub weights: Vec<f64>,
    learning_rate: f64,
    max_iterations: usize,
    verbose: bool,
}

impl LinearClassification {
    pub fn new(learning_rate: Option<f64>, max_iterations: Option<usize>, verbose: Option<bool>) -> Self {
        Self {
            weights: Vec::new(),
            learning_rate: learning_rate.unwrap_or(0.01),
            max_iterations: max_iterations.unwrap_or(1000),
            verbose: verbose.unwrap_or(false),
        }
    }
    fn add_bias(x: &[f64]) -> Vec<f64> {
        let mut v = Vec::with_capacity(x.len() + 1);
        v.push(1.0);
        v.extend(x.iter().cloned());
        v
    }

    fn predict_fit(&self, x_aug: &[f64]) -> f64 {
        let activation: f64 = x_aug.iter()
            .zip(self.weights.iter())
            .map(|(x, w)| w * x)
            .sum();
        if activation > 0.0 { 1.0 } else { -1.0 }
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        let x_aug = Self::add_bias(x);
        self.predict_fit(&x_aug)
    }

    pub fn fit(&mut self, x_data: &[Vec<f64>], y_data: &[f64]) {
        if x_data.len() != y_data.len() {
            panic!("Error: x and y must have the same length.");
        }

        let x_data_bias: Vec<Vec<f64>> = x_data.iter()
            .map(|x_i| Self::add_bias(x_i))
            .collect();

        let n = x_data_bias[0].len();
        self.weights = vec![0.0; n];

        let mut errors= 0;

        for iter in 0..self.max_iterations {
            errors = 0;
            for (x, &y) in x_data_bias.iter().zip(y_data.iter()) {
                let y_pred = self.predict_fit(x);
                if y_pred != y {
                    self.weights = self.weights.iter()
                        .zip(x.iter())
                        .map(|(w_i, x_i)| w_i + self.learning_rate * (y - y_pred) * x_i)
                        .collect::<Vec<f64>>();
                    errors += 1;
                }
            }
            if errors == 0 {
                if self.verbose {
                    println!("Convergence reached after {} iterations.", iter);
                }
                break;
            }
        }
        if errors != 0 && self.verbose {
            println!("Maximum number of iterations reached, convergence not reached.")
        }
    }
    
    pub fn get_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }
    
    pub fn get_bias(&self) -> f64 {
        if self.weights.is_empty() {
            0.0
        } else {
            self.weights[0]
        }
    }
}

// #[derive(Serialize, Deserialize, Debug, Clone)]
// pub struct LinearRegression {
//     pub weights: Option<Vec<f64>>,
//     pub bias: Option<f64>,
// }

// impl LinearRegression {
//     pub fn new() -> Self {
//         Self { weights: None, bias: None }
//     }

//     pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
//         let n_samples = x.len();
//         let n_features = if n_samples > 0 { x[0].len() } else { 0 };

//         let flat_x: Vec<f64> = x.iter().flat_map(|row| row.iter()).copied().collect();
//         let x_arr = Array2::from_shape_vec((n_samples, n_features), flat_x)
//             .expect("Bad input shape for X");
//         let y_arr = Array1::from(y.to_vec());

//         let x_aug = Self::add_bias(&x_arr);

//         let xtx = x_aug.t().dot(&x_aug);
//         let xty = x_aug.t().dot(&y_arr);
//         let xtx_inv = xtx.inv().expect("XTX not inversible !");
//         let all_weights = xtx_inv.dot(&xty);

//         self.bias = Some(all_weights[0]);
//         self.weights = Some(all_weights.slice(s![1..]).to_vec());
//     }

//     pub fn predict(&self, x: &[f64]) -> f64 {
//         let weights = self.weights.as_ref().expect("Model not fit");
//         let bias = self.bias.expect("Model not fit");
//         x.iter().zip(weights.iter()).map(|(xi, wi)| xi * wi).sum::<f64>() + bias
//     }

//     fn add_bias(x: &Array2<f64>) -> Array2<f64> {
//         let n_samples = x.nrows();
//         let n_features = x.ncols();
//         let mut x_aug = Array2::<f64>::ones((n_samples, n_features + 1));
//         for i in 0..n_samples {
//             for j in 0..n_features {
//                 x_aug[(i, j + 1)] = x[(i, j)];
//             }
//         }
//         x_aug
//     }

//     pub fn save_json(&self, path: &str) -> std::io::Result<()> {
//         let file = File::create(path)?;
//         serde_json::to_writer_pretty(file, self)?;
//         Ok(())
//     }

//     pub fn load_json(path: &str) -> std::io::Result<Self> {
//         let file = File::open(path)?;
//         let model = serde_json::from_reader(file)?;
//         Ok(model)
//     }
// }
use rand::prelude::SliceRandom;
use serde::{Deserialize, Serialize};
use crate::kmeans::KMeans;
use crate::utils::{distance, transpose, matrix_multiply, invert_matrix, matrix_vector_product};

/// Sigmoid for classification.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}


/// RBF using k-means centroids as centers
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RBFKMeans {

    centers: Vec<Vec<f64>>,
    weights: Vec<f64>,
    gamma: f64,
    is_class: bool,
}

impl RBFKMeans {
    pub fn new(
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        k: usize,
        gamma: f64,
        max_iters: usize,
        is_class: bool,
    ) -> Self {
        // 1. fit k-means
        let centroids = KMeans::new(k, max_iters).fit(&x);
        // 2. build design matrix Phi (m x k)
        let m = x.len();
        let mut phi = vec![vec![0.0; k]; m];
        for i in 0..m {
            for j in 0..k {
                phi[i][j] = (-gamma * distance(&x[i], &centroids[j])).exp();
            }
        }
        // 3. compute W = (Phi^T Phi)^-1 Phi^T Y
        let phi_t = transpose(&phi);            // k x m
        let phi_t_phi = matrix_multiply(&phi_t, &phi); // k x k
        let inv = invert_matrix(&phi_t_phi);   // k x k
        let phi_t_y = matrix_vector_product(&phi_t, &y); // k
        let w = matrix_vector_product(&inv, &phi_t_y);

        Self {
            centers: centroids,
            weights: w,
            gamma: gamma,
            is_class: is_class,
        }
    }

    pub fn predict(&self, x_new: &[f64]) -> f64 {
        let k = self.centers.len();
        let mut out = 0.0;
        for j in 0..k {
            let basis = (-self.gamma * distance(x_new, &self.centers[j])).exp();
            out += self.weights[j] * basis;
        }
        if self.is_class {
            sigmoid(out)
        } else {
            out
        }
    }
}

// Example usage:
// fn main() -> Result<(), Box<dyn Error>> {
//     let X = vec![vec![1.0,4.0], vec![0.5,0.5], vec![2.0, 3.0]];
//     let Y = vec![1.0,0.0,1.0];
//     let model = RBFKMeans::new(
//         &X,
//         &Y,
//         3, // number of clusters
//         0.5, // gamma
//         100, // max iterations
//         true, // is classification
//     );
//     let pred = model.predict(&[0.0,0.0]);
//     println!("Prediction rbf: {}", pred);
//     Ok(())
// }

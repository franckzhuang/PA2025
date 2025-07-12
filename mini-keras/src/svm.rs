use osqp::{CscMatrix, Problem, Settings}; // QP solver for dual optimization
use std::f64::INFINITY;
use serde::{Deserialize, Serialize};

/// Kernel
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum KernelType {
    Linear, // kernel type | 'linear' or 'rbf'
    RBF(f64), // RBF kernel parameter
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SVM {
    pub c: Option<f64>,              // regularization | None -> hard margin | >0 -> soft margin
    pub kernel: KernelType,          // kernel type | 'linear' or 'rbf'

    pub alphas: Vec<f64>,            // optimal solutions from QP (Coeff Dual)
    pub support_vectors: Vec<Vec<f64>>, // X_i with α_i > 0
    pub support_labels: Vec<f64>,    //  y_i

    pub b: f64,                      // Bias
}

fn linear_kernel(x: &[f64], y: &[f64]) -> f64 {
    // K(x,y) = x^T y
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

fn rbf_kernel(x: &[f64], y: &[f64], gamma: f64) -> f64 {
    // K(x,y) = exp(-γ ||x - y||²) (Vectorized formula from Course - Gaussian)
    let squared_norm: f64 = x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum();
    (-gamma * squared_norm).exp()
}

fn kernel(x: &[f64], y: &[f64], k: &KernelType) -> f64 {
    match k {
        KernelType::Linear => linear_kernel(x, y),
        KernelType::RBF(gamma) => rbf_kernel(x, y, *gamma),
    }
}

impl SVM {
    /// __init__
    pub fn new(c: Option<f64>, kernel: KernelType) -> Self {
        Self {
            c,
            kernel,
            alphas: vec![],
            support_vectors: vec![],
            support_labels: vec![],
            b: 0.0,
        }
    }

    /// Minimize (1/2) α^T P α + q^T α
    /// subject to G α ≤ h, A α = b

    /// where:
    /// - P[i,j] = y_i y_j K(x_i, x_j) (Gram matrix with kernel)
    /// - q = vector of -1
    /// - A = row vector of labels y^T (equality constraint)
    /// - b = 0
    /// - G and h encode constraints 0 ≤ α_i ≤ C (soft margin) or α_i ≥ 0 (hard margin)
    pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>) {
        
        let n = x.len();

        // P[i][j] = y_i y_j K(x_i, x_j)
        // Flatten P to a 1D list for osqp | compatible if P symmetric
        let mut p_dense = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                p_dense[j * n + i] = y[i] * y[j] * kernel(&x[i], &x[j], &self.kernel);
            }
        }
        let p = CscMatrix::from_column_iter(n, n, p_dense).into_upper_tri();

        // linear term
        let q = vec![-1.0; n];

        // Equality constraint (A, l, u)
        // Constraint 1 : y^T α = 0
        let mut a_mat = vec![vec![0.0; n]; n + 1];
        for j in 0..n {
            a_mat[0][j] = y[j];
            a_mat[j + 1][j] = 1.0;
        }
        let a = CscMatrix::from_row_iter(
            n + 1,
            n,
            a_mat.iter().flat_map(|row| row.iter().cloned()).collect::<Vec<_>>()
        );

        // Contraint 2 : 0 ≤ α_i ≤ C (soft margin) or α_i ≥ 0 (hard margin)

        let c = self.c.unwrap_or(INFINITY);
        let l = vec![0.0; n + 1]; // Lower bounds (always 0)
        let mut u = vec![0.0];        // Upper bounds
        u.extend(vec![c; n]);         // Bounds if α_i (C or ∞)

        // OSQP Parameters
        let settings = Settings::default()
            .max_iter(20_000)
            .eps_abs(1e-4)
            .eps_rel(1e-4)
            .verbose(false); // Solver Output

        // QP Solver
        let mut prob = Problem::new(
            p,
            &q,
            a,
            &l,
            &u,
            &settings,
        ).expect("Error : OSQP Build failed");

        let result = prob.solve();
        let solution = match result.x() {
            Some(sol) => sol,
            None => {
                panic!(
                    "OSQP did not found any solution (statut {:?})",
                    result
                )
            }
        };

        // Select support vectors with α_i > 0 | 1e-5 to avoid numerical rounding
        self.alphas = solution.iter().cloned().collect();
        self.support_vectors.clear();
        self.support_labels.clear();
        let threshold = 1e-5; // tolérance pour la sélection des supports
        for (i, &alpha) in self.alphas.iter().enumerate() {
            if alpha > threshold {
                self.support_vectors.push(x[i].clone());
                self.support_labels.push(y[i]);
            }
        }

        // y_i ( sum_j α_j y_j K(x_j, x_i) + b ) = 1
        // b = y_i - sum_j α_j y_j K(x_j, x_i)        self.b = 0.0;
        let m = self.support_vectors.len();
        for i in 0..m {
            let mut sum = 0.0;
            for j in 0..m {
                sum += self.alphas[j] * self.support_labels[j]
                    * kernel(&self.support_vectors[j], &self.support_vectors[i], &self.kernel);
            }
            self.b += self.support_labels[i] - sum;
        }
        if m > 0 {
            self.b /= m as f64;
        }
    }

    pub fn weights(&self) -> Option<Vec<f64>> {
        if let KernelType::Linear = self.kernel {
            if self.support_vectors.is_empty() {
                return Some(vec![]);
            }
    
            let mut w = vec![0.0; self.support_vectors[0].len()];
            for i in 0..self.support_vectors.len() {
                for j in 0..w.len() {
                    w[j] += self.alphas[i]
                        * self.support_labels[i]
                        * self.support_vectors[i][j];
                }
            }
            Some(w)
        } else {
            None
        }
    }

    pub fn project(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
        // f(x) = ∑_{j} α_j y_j K(x_j, x) + b
        // Decision score for each point
        x.iter()
            .map(|xi| {
                let mut sum = 0.0;
                for j in 0..self.support_vectors.len() {
                    sum += self.alphas[j] * self.support_labels[j]
                        * kernel(&self.support_vectors[j], xi, &self.kernel);
                }
                sum + self.b
            })
            .collect()
    }

    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<i8> {
        self.project(x)
            .iter()
            .map(|&val| if val >= 0.0 { 1 } else { -1 })
            .collect()
    }    
}

fn main() {
    // ===== Dataset 1D =====
    let x_train_1d: Vec<Vec<f64>> = (-5..=5).map(|x| vec![x as f64]).collect();
    let y_train_1d: Vec<f64> = x_train_1d.iter().map(|x| if x[0] < 0.0 { -1.0 } else { 1.0 }).collect();

    // ===== Dataset 2D =====
    let mut x_train_2d = Vec::new();
    for x in -3..=3 {
        for y in -3..=3 {
            x_train_2d.push(vec![x as f64, y as f64]);
        }
    }
    let y_train_2d: Vec<f64> = x_train_2d.iter()
        .map(|point| if point[0] + point[1] >= 0.0 { 1.0 } else { -1.0 })
        .collect();

    // ===== Dataset 3D =====
    let mut x_train_3d = Vec::new();
    for x in -2..=2 {
        for y in -2..=2 {
            for z in -2..=2 {
                x_train_3d.push(vec![x as f64, y as f64, z as f64]);
            }
        }
    }
    let y_train_3d: Vec<f64> = x_train_3d.iter()
        .map(|point| if point.iter().sum::<f64>() >= 0.0 { 1.0 } else { -1.0 })
        .collect();

    // === SVM Hard margin (en vrai C=1e6 pour approximer "hard") ===
    let mut svm = SVM::new(Some(1e6), KernelType::Linear);

    // ---------- Test 1D ----------
    println!("==== Test 1D ====");
    svm.fit(&x_train_1d, &y_train_1d);
    let preds_1d = svm.predict(&x_train_1d);
    println!("X_train_1d: {:?}", x_train_1d.iter().map(|x| x[0]).collect::<Vec<_>>());
    println!("y_train_1d: {:?}", y_train_1d);
    println!("predictions: {:?}", preds_1d);
    println!();

    // ---------- Test 2D ----------
    println!("==== Test 2D ====");
    svm.fit(&x_train_2d, &y_train_2d);
    let preds_2d = svm.predict(&x_train_2d);
    println!("Sample 2D points and predictions:");
    let step = std::cmp::max(1, x_train_2d.len() / 10);
    for i in (0..x_train_2d.len()).step_by(step) {
        println!(
            "Point: {:?}, label: {}, pred: {}",
            x_train_2d[i], y_train_2d[i], preds_2d[i]
        );
    }
    println!();

    // ---------- Test 3D ----------
    println!("==== Test 3D ====");
    svm.fit(&x_train_3d, &y_train_3d);
    let preds_3d = svm.predict(&x_train_3d);
    println!("Sample 3D points and predictions:");
    let step = std::cmp::max(1, x_train_3d.len() / 10);
    for i in (0..x_train_3d.len()).step_by(step) {
        println!(
            "Point: {:?}, label: {}, pred: {}",
            x_train_3d[i], y_train_3d[i], preds_3d[i]
        );
    }
}
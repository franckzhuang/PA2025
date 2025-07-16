
use crate::utils::{distance};

/// K-means clustering
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KMeans {
    pub k: usize,
    pub max_iters: usize,
}

impl KMeans {
    pub fn new(k: usize, max_iters: usize) -> Self {
        Self { k, max_iters }
    }

    pub fn fit(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut centroids = self.initialize_centroids(x);
        let mut labels = vec![0; x.len()];

        for _ in 0..self.max_iters {
            // assign
            for (i, xi) in x.iter().enumerate() {
                labels[i] = centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let da = distance(xi, a);
                        let db = distance(xi, b);
                        da.partial_cmp(&db).unwrap()
                    })
                    .map(|(idx, _)| idx)
                    .unwrap();
            }
            // update
            let mut counts = vec![0; self.k];
            let dim = x[0].len();
            let mut sums = vec![vec![0.0; dim]; self.k];
            for (xi, &lbl) in x.iter().zip(labels.iter()) {
                counts[lbl] += 1;
                for d in 0..dim {
                    sums[lbl][d] += xi[d];
                }
            }
            let mut converged = true;
            for i in 0..self.k {
                if counts[i] > 0 {
                    for d in 0..dim {
                        let new_c = sums[i][d] / counts[i] as f64;
                        if (new_c - centroids[i][d]).abs() > 1e-6 {
                            converged = false;
                        }
                        centroids[i][d] = new_c;
                    }
                }
            }
            if converged {
                break;
            }
        }
        centroids
    }

    fn initialize_centroids(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut rng = rand::rng();
        let mut centroids = x.to_vec();
        centroids.shuffle(&mut rng);
        centroids.truncate(self.k);
        centroids
    }
}
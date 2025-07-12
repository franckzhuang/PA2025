
use rand::prelude::IndexedRandom;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Kmeans {
    pub num_clusters: usize,
    pub num_iterations: usize,
    pub max_iter: usize,
    pub tol: f64,
}


impl Kmeans {
    pub fn new(num_clusters: usize, num_iterations: usize, max_iter: usize, tol: f64) -> Self {
        Self {
            num_clusters,
            num_iterations,
            max_iter,
            tol,
        }
    }

    pub fn initialize_centroids(&self, X: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut centroids = Vec::new();
        let mut rng = rand::thread_rng();
        let indices: Vec<usize> = (0..X.len()).collect();
        let selected_indices = indices.choose_multiple(&mut rng, self.num_clusters);

        for &index in selected_indices {
            centroids.push(X[index].clone());
        }

        centroids
    }

    pub fn assign_clusters(&self, X: &[Vec<f64>], centroids: &[Vec<f64>]) -> Vec<usize> {
        let mut labels = vec![0; X.len()];

        for (i, x) in X.iter().enumerate() {
            let mut min_distance = f64::MAX;
            let mut closest_centroid = 0;

            for (j, centroid) in centroids.iter().enumerate() {

                // distance  = ||x - centroid||^2
                let distance = x.iter()
                    .zip(centroid.iter())
                    .map(|(xi, ci)| (xi - ci).powi(2))
                    .sum::<f64>();


                if distance < min_distance {
                    min_distance = distance;
                    closest_centroid = j;
                }
            }

            labels[i] = closest_centroid;
        }

        labels
    }

    pub fn update_centroids(&self, X: &[Vec<f64>], labels: &[usize]) -> Vec<Vec<f64>> {
        let mut centroids = vec![vec![0.0; X[0].len()]; self.num_clusters];
        let mut counts = vec![0; self.num_clusters];
        // goal is to average the coordinates of the points in each cluster

        // sum here is the first part
        for (i, x) in X.iter().enumerate() {
            let label = labels[i];
            counts[label] += 1;

            for (j, &value) in x.iter().enumerate() {
                centroids[label][j] += value;
            }
        }

        // average here is the second part by dividing by the number of points in each cluster
        for (i, centroid) in centroids.iter_mut().enumerate() {
            if counts[i] > 0 {
                for value in centroid.iter_mut() {
                    *value /= counts[i] as f64;
                }
            }
        }

        centroids // new centroids
    }

    pub fn has_converged(&self, centroids: &[Vec<f64>]) -> bool {
        return false;
    }

    pub fn fit(&mut self, X: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut centroids = self.initialize_centroids(X);
        let mut labels = vec![0; X.len()];

        for _ in 0..self.num_iterations {
            labels = self.assign_clusters(X, &centroids);
            centroids = self.update_centroids(X, &labels);

            if self.has_converged(&centroids) {
                break;
            }
        }

        centroids
    }


}

pub fn main() {
    let mut kmeans = Kmeans::new(2, 100, 300, 0.001);
    let data = vec![
        vec![-1.0],
        vec![-3.0],
        vec![-5.0],
        vec![1.0],
        vec![2.0],
        vec![3.0],
    ];

    let centroids  = kmeans.fit(&data);

    println!("fitted centroids: {:?}", centroids);
}
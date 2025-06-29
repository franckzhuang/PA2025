use std::f64;
use rand::Rng;

/// returns a random f64 in [-1.0, 1.0]
fn random() -> f64 {
    let mut rng = rand::rng();
    rng.random_range(-1.0..=1.0)
}

/// A single perceptron (neuron) with weights, bias, and cached state
pub struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
    last_inputs: Vec<f64>,
    last_z: f64,
}

impl Perceptron {
    /// Create a new perceptron with `inputs` incoming connections
    pub fn new(inputs: usize) -> Self {
        let weights = (0..inputs).map(|_| random()).collect();
        let bias = random();
        Perceptron {
            weights,
            bias,
            last_inputs: Vec::new(),
            last_z: 0.0,
        }
    }

    /// Compute pre-activation z = w*x + b, cache inputs x and outputs z
    pub fn predict(&mut self, inputs: &[f64]) -> f64 {
        assert_eq!(inputs.len(), self.weights.len(), "Incorrect input size");
        self.last_inputs = inputs.to_vec();
        let z = self.weights.iter()
            .zip(inputs.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + self.bias;
        self.last_z = z;
        z
    }

    /// Update weights and bias given gradient dL/dz and learning rate
    pub fn update(&mut self, delta_z: f64, lr: f64) {
        for (w, &x) in self.weights.iter_mut().zip(self.last_inputs.iter()) {
            *w -= lr * delta_z * x;
        }
        self.bias -= lr * delta_z;
    }
}

/// A fully-connected layer consisting of multiple perceptrons
pub struct Dense {
    neurons: Vec<Perceptron>,
    activation: Activation,
    last_inputs: Vec<f64>,
    last_zs: Vec<f64>,
    last_activations: Vec<f64>,
}

/// Supported activation functions
pub enum Activation {
    Sigmoid,
    Linear,
}

impl Dense {
    /// Create a new Dense layer
    pub fn new(num_inputs: usize, num_neurons: usize, activation: Activation) -> Self {
        let neurons = (0..num_neurons)
            .map(|_| Perceptron::new(num_inputs))
            .collect();
        Dense {
            neurons,
            activation,
            last_inputs: Vec::new(),
            last_zs: Vec::new(),
            last_activations: Vec::new(),
        }
    }

    /// Sigmoid function
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Derivative of sigmoid at x (in terms of pre-activation value)
    fn sigmoid_derivative(x: f64) -> f64 {
        let s = Self::sigmoid(x);
        s * (1.0 - s)
    }

    /// Forward pass: compute activations from inputs
    pub fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        self.last_inputs = inputs.to_vec();
        let zs: Vec<f64> = self.neurons.iter_mut()
            .map(|neuron| neuron.predict(inputs))
            .collect();
        self.last_zs = zs.clone();

        let activations: Vec<f64> = match self.activation {
            Activation::Sigmoid => zs.iter().map(|&z| Self::sigmoid(z)).collect(),
            Activation::Linear => zs.clone(),
        };
        self.last_activations = activations.clone();
        activations
    }

    /// Backward pass: update weights & biases, return deltas for previous layer
    pub fn backward(&mut self, deltas: &[f64], lr: f64) -> Vec<f64> {
        // compute gradient w.r.t. pre-activation
        let delta_zs: Vec<f64> = deltas.iter().enumerate().map(|(j, &d)| {
            match self.activation {
                Activation::Sigmoid => d * Self::sigmoid_derivative(self.last_zs[j]),
                Activation::Linear => d,
            }
        }).collect();

        // prepare vector for accumulating previous layer deltas
        let mut prev_deltas = vec![0.0; self.last_inputs.len()];

        // update each neuron and accumulate
        for (j, neuron) in self.neurons.iter_mut().enumerate() {
            let dz = delta_zs[j];
            neuron.update(dz, lr);
            for (k, &w) in neuron.weights.iter().enumerate() {
                prev_deltas[k] += dz * w;
            }
        }
        prev_deltas
    }
}

/// A simple multi-layer perceptron
pub struct MLP {
    layers: Vec<Dense>,
    is_classification: bool,
}

impl MLP {
    /// Create a new MLP from a list of layers
    pub fn new(mut layers: Vec<Dense>, is_classification: bool) -> Self {
        // if classification :
        // ensure last layer has 1 neuron
        // and has sigmoid activation
        // if not add a Dense layer with 1 neuron and sigmoid activation
        if is_classification && !layers.is_empty() {
            let last_layer = layers.last().unwrap();
            if last_layer.neurons.len() != 1 || matches!(last_layer.activation, Activation::Linear) {
                // let mut new_layers = layers;
                layers.push(Dense::new(last_layer.neurons.len(), 1, Activation::Sigmoid));
                return MLP { layers, is_classification: true };
            }
        }


        MLP { layers, is_classification }
    }

    /// Predict outputs for a single input sample
    pub fn predict(&mut self, inputs: &[f64]) -> Vec<f64> {
        let mut activations = inputs.to_vec();
        for layer in self.layers.iter_mut() {
            activations = layer.forward(&activations);
        }
        activations
    }

    /// Train on dataset x and targets y using SGD and MSE loss
    pub fn train(&mut self,
                 x: &[Vec<f64>],
                 y: &[f64],
                 epochs: usize,
                 lr: f64
    ) {
        for _ in 0..epochs {
            for (xi, &yi) in x.iter().zip(y.iter()) {
                // forward pass
                let mut activations = xi.clone();
                for layer in self.layers.iter_mut() {
                    activations = layer.forward(&activations);
                }
                // compute MSE gradient
                let outs = &activations;
                let deltas: Vec<f64> = outs.iter().map(|&out| 2.0 * (out - yi)).collect();
                // backward pass
                let mut deltas = deltas;
                for layer in self.layers.iter_mut().rev() {
                    deltas = layer.backward(&deltas, lr);
                }
            }
        }
    }
}

// Example usage (XOR)
fn main() {
    let l1 = Dense::new(2, 2, Activation::Sigmoid);
    let l2 = Dense::new(2, 1, Activation::Linear);
    let mut mlp = MLP::new(vec![l1, l2], true);
    let x = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0], vec![1.0, 1.0]];
    let y = vec![1.0,1.0,0.0,0.0];
    mlp.train(&x, &y, 100_000, 0.01);
    for x in x.iter() {
        println!("Prediction for {:?}: {:?}", x, mlp.predict(x));
    }
}

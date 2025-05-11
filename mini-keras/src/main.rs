mod linear_model;
mod tests;
mod utils;

use rand::Rng;

pub fn random_() -> f64 {
    let mut rng = rand::thread_rng();
    rng.gen_range(-1.0..=1.0)
}

pub enum Prediction {
    Classification(f64),
    Regression(Vec<f64>),
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub struct Perceptron {
    pub weights: Vec<f64>,
    pub bias: f64
}

pub struct DenseLayer {
    pub neurons: Vec<Perceptron>,
}

pub struct MLP {
    pub layers: Vec<DenseLayer>,
    pub is_classification: bool,
}


impl Perceptron {

    pub fn new(inputs: usize) -> Self {
        Self {
            weights: (0..inputs).map(|_| random_()).collect(),
            bias: random_()
        }
    }

    pub fn predict(&self, inputs: &[f64]) -> f64 {
        let sum: f64 = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, i)| w * i)
            .sum();
        sum + self.bias
    }

    pub fn update(&mut self, inputs: &[f64], delta: f64, learning_rate: f64) {
        for (weight, input) in self.weights.iter_mut().zip(inputs.iter()) {
            *weight -= learning_rate * delta * input;
        }
        self.bias -= delta * learning_rate;
    }

}



impl DenseLayer {
    pub fn new(num_inputs: usize, num_neurons: usize) -> Self {
        Self {
            neurons: (0..num_neurons)
                .map(|_| Perceptron::new(num_inputs))
                .collect(),
        }
    }

    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons.iter().map(|neuron| neuron.predict(inputs)).collect()
    }
}


impl MLP {
    pub fn new(layers: Vec<DenseLayer>, is_classification : bool) -> Self {
        Self {
            layers,
            is_classification,
        }
    }

    pub fn predict(&self, inputs: &[f64]) -> Prediction {
        let mut outputs = inputs.to_vec(); // will hold intermediate results
        for layer in &self.layers {
            outputs = layer.forward(&outputs);
        }



        if self.is_classification {
            if outputs.len() == 1 {
                return Prediction::Classification(sigmoid(outputs[0]));
            } else {
                // raise an error
                panic!("cannot do classification if output size is not 1")
            }
        }
        // regression
        Prediction::Regression(outputs)
    }


    pub fn train(
        &mut self,
        X: &[Vec<f64>],
        y: &[f64],
        epochs: usize,
        lr: f64,
    ) {
        for _ in 0..epochs {
            for (xi, &yi) in X.iter().zip(y.iter()) {
                // forward
                let mut layer_inputs: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len() + 1);
                layer_inputs.push(xi.clone());
                let mut pre_activations: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len());
                let mut activations = xi.clone();
                for layer in &self.layers {
                    let zs = layer.forward(&activations);
                    pre_activations.push(zs.clone());
                    activations = zs;
                    layer_inputs.push(activations.clone());
                }
                // compute output delta
                let mut deltas: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len());
                if self.is_classification {
                    let z = pre_activations.last().unwrap()[0];
                    let a = sigmoid(z);
                    let delta = 2.0 * (a - yi) * sigmoid_derivative(z);
                    deltas.push(vec![delta]);
                } else {
                    let z = pre_activations.last().unwrap();
                    let delta_vec: Vec<f64> = z.iter().map(|&a| 2.0 * (a - yi)).collect();
                    deltas.push(delta_vec);
                }
                // backprop hidden layers
                for l in (1..self.layers.len()).rev() {
                    let layer = &self.layers[l];
                    let prev_len = self.layers[l - 1].neurons.len();
                    let mut next_delta = vec![0.0; prev_len];
                    for k in 0..prev_len {
                        let mut error = 0.0;
                        for (j, neuron) in layer.neurons.iter().enumerate() {
                            error += deltas.last().unwrap()[j] * neuron.weights[k];
                        }
                        next_delta[k] = error; // identity activation derivative = 1
                    }
                    deltas.push(next_delta);
                }
                deltas.reverse();
                // update weights
                for (l, layer) in self.layers.iter_mut().enumerate() {
                    let inputs_to_layer = &layer_inputs[l];
                    for (j, neuron) in layer.neurons.iter_mut().enumerate() {
                        let d = deltas[l][j];
                        neuron.update(inputs_to_layer, d, lr);
                    }
                }
            }
        }
    }


}

fn main() {

    // RNG test predict
    // println!("{}", random_())

    // sigmoid test predict
    // println!("{}", sigmoid(1.0));
    // println!("{}", sigmoid(0.0));
    // println!("{}", sigmoid(0.5));
    // println!("{}", sigmoid(100.0));
    // println!("{}", sigmoid(-100.0));
    //
    // println!("{}", sigmoid_derivative(1.0));
    // println!("{}", sigmoid_derivative(100.0));
    // println!("{}", sigmoid_derivative(-100.0));


    // Perceptron test predict
    // let perceptron = Perceptron::new(4);
    // let inputs = vec![1.0, 3.1, 2.0, 7.4];
    // let pred = perceptron.predict(&inputs);
    // println!("{}", pred);

    // Dense test predict
    // let dense_layer = DenseLayer::new(10, 10);
    // let inputs = vec![0.2, 0.4, 0.6, 0.8, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    // let outputs = dense_layer.forward(&inputs);
    // // print each number in outputs //
    // println!("{:?}", outputs);

    // MLP test predict
    // let layer1 = DenseLayer::new(3, 10);
    // let layer2 = DenseLayer::new(10, 28);
    // let layer3 = DenseLayer::new(28, 1);
    //
    // let layers = vec![layer1, layer2, layer3];
    //
    // let mlp = MLP::new(layers, true);
    //
    // let inputs = vec![0.2, 0.4, 0.6];
    // let outputs = mlp.predict(&inputs);
    // // print each number in outputs //
    // match outputs {
    //     Prediction::Classification(value) => println!("Classification: {}", value),
    //     Prediction::Regression(values) => println!("Regression: {:?}", values),
    // }

    // Regression: learn y = x identity
    println!("Regression Identity example");
    let mut l1 = DenseLayer::new(1, 4);
    let mut l2 = DenseLayer::new(4, 1);
    let mut mlp = MLP::new(vec![l1, l2], false);
    let X = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    println!("before training");
    for (input, &target) in X.iter().zip(y.iter()) {
        let Prediction::Regression(outputs) = mlp.predict(input) else {panic!("unknown error")};
        println!("{:?}", outputs)
    }
    mlp.train(&X, &y, 5000, 0.01);
    println!("after training");
    for (input, &target) in X.iter().zip(y.iter()) {
        let Prediction::Regression(outputs) = mlp.predict(input) else {panic!("unknown error")};
        println!("{:?}", outputs)
    }

    // Classification test
    // Classification: threshold at x>=2
    println!("Classification example:  1 if x>= 2 else 0");
    let mut l1 = DenseLayer::new(1, 2);
    let mut l2 = DenseLayer::new(2, 1);
    let mut mlp = MLP::new(vec![l1, l2], true);
    let X = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let y = vec![0.0, 0.0, 1.0, 1.0];
    println!("before training");
    for (input, &target) in X.iter().zip(y.iter()) {
        let Prediction::Classification(prob) = mlp.predict(input) else {panic!("unknow error")};
        println!("{:?}", prob)
    }
    mlp.train(&X, &y, 5000, 0.1);
    println!("after training");
    for (input, &target) in X.iter().zip(y.iter()) {
        let Prediction::Classification(prob) = mlp.predict(input) else {panic!("unknow error")};
        println!("{:?}", prob)
    }




}
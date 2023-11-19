use candle_core::{Tensor, Error};
use candle_nn;

pub struct GenericLayer {
    layer: Box<dyn candle_nn::Module>,
}

pub struct ModelBuilder {
    layers: Vec<GenericLayer>,
}

impl ModelBuilder {
    pub fn new() -> Self {
        ModelBuilder {
            layers: Vec::new(),
        }
    }

    pub fn add_layer<T: 'static + candle_nn::Module>(mut self, layer: T) -> Self {
        self.layers.push(GenericLayer { layer: Box::new(layer) });
        self
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.layer.forward(&output)?;
        }
        Ok(output)
    }

    // Method to build/compile the model
}

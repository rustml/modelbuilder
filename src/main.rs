use candle_nn::{Linear, LayerNorm, Module};
use candle_core::{Tensor, Device::Cpu};
mod lib;
use lib::{ModelBuilder, GenericLayer};

fn main() -> candle_core::Result<()> {
    // Create the ModelBuilder
    let mut model_builder = ModelBuilder::new();

    // Add layers to the ModelBuilder
    model_builder.add_layer(Linear::new(
        Tensor::new(&[[1f32, 2.], [3., 4.]], &Cpu)?, 
        Some(Tensor::new(&[0.5, 1.0], &Cpu)?)
    ));
    model_builder.add_layer(LayerNorm::new(
        Tensor::new(1f32, &Cpu)?, 
        Tensor::new(0f32, &Cpu)?, 
        1e-5
    ));

    // Sample input tensor
    let input = Tensor::new(&[[0.5, 1.5]], &Cpu)?;

    // Use the ModelBuilder's forward method
    let final_output = model_builder.forward(&input)?;

    println!("Output: {:?}", final_output);
    Ok(())
}
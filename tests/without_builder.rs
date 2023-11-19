use candle_core::{Tensor, Error, Module};
use candle_nn::{Linear, LayerNorm};
use candle_core::Device::Cpu;

#[test]
fn test_model_without_builder() -> Result<(), Error> {
    let linear_layer = Linear::new(
        Tensor::new(&[[1f32, 2f32], [3f32, 4f32]], &Cpu)?, 
        Some(Tensor::new(&[0.5f32, 1.0f32], &Cpu)?)
    );
    let layer_norm = LayerNorm::new(
        Tensor::new(1f32, &Cpu)?, 
        Tensor::new(0f32, &Cpu)?, 
        1e-5f64
    );

    let input = Tensor::new(&[[0.5f32, 1.5f32]], &Cpu)?;

    let linear_output = linear_layer.forward(&input)?;
    let final_output = layer_norm.forward(&linear_output)?;

    // Check if the output tensor has the expected number of dimensions
    let output_dims = final_output.dims();
    assert_eq!(output_dims.len(), 2); // Assuming the output is 2D as well
    Ok(())
}

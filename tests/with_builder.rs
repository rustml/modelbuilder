use candle_core::{Tensor, Error};
use modelbuilder::ModelBuilder;
use candle_nn::{Linear, LayerNorm};
use candle_core::Device::Cpu;

#[test]
fn test_model_with_builder() -> Result<(), Error> {
    let mut model_builder = ModelBuilder::new()
        .add_layer(Linear::new(
            Tensor::new(&[[1.0, 2.0], [3.0, 4.0]], &Cpu)?, 
            Some(Tensor::new(&[0.5, 1.0], &Cpu)?)
        ))
        .add_layer(LayerNorm::new(
            Tensor::new(1.0, &Cpu)?, 
            Tensor::new(0.0, &Cpu)?, 
            1e-5
        ));

    let input = Tensor::new(&[[0.5, 1.5]], &Cpu)?;

    let final_output = model_builder.forward(&input)?;

    let output_dims = final_output.dims();
    assert_eq!(output_dims.len(), 2);
    Ok(())
}

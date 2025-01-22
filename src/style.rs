use ndarray::{s, Array1, Array2};
use serde::Deserialize;

use crate::errors::Sbv2CoreError;

#[derive(Deserialize)]
pub struct Data {
    pub shape: [usize; 2],
    pub data: Vec<Vec<f32>>,
}

pub fn load_style<T>(style: T) -> Result<Array2<f32>, Sbv2CoreError>
where
    T: AsRef<[u8]>,
{
    let data: Data = serde_json::from_slice(style.as_ref())?;

    Ok(Array2::from_shape_vec(
        data.shape,
        data.data.iter().flatten().copied().collect(),
    )?)
}

pub fn get_style_vector(
    style_vectors: &Array2<f32>,
    style_id: i32,
    weight: f32,
) -> Result<Array1<f32>, Sbv2CoreError> {
    let mean = style_vectors.slice(s![0, ..]).to_owned();
    let style_vector = style_vectors.slice(s![style_id as usize, ..]).to_owned();
    let diff = (style_vector - &mean) * weight;

    Ok(mean + &diff)
}

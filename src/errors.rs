use thiserror::Error;

#[derive(Error, Debug)]
pub enum Sbv2CoreError {
    #[error("model not found error")]
    ModelNotFoundError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Ort error: {0}")]
    OrtError(#[from] ort::Error),

    #[error("Tokenizers error: {0}")]
    TokenizersError(#[from] tokenizers::Error),

    #[error("serde_json error: {0}")]
    SerdeJsonError(#[from] serde_json::Error),

    #[error("NdArray error: {0}")]
    NdArrayError(#[from] ndarray::ShapeError),

    #[error("JPreprocess error: {0}")]
    JPreprocessError(#[from] jpreprocess::error::JPreprocessError),

    #[error("Value error: {0}")]
    ValueError(String),

    #[error("hound error: {0}")]
    HoundError(#[from] hound::Error),
}

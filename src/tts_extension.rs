use std::path::PathBuf;

use crate::{errors::Sbv2CoreError, TtsModelHolder};

pub trait TtsModelHolderFromPath {
    fn load_from_path<P>(
        &mut self,
        model_ident: &str,
        style_vectors_path: P,
        vits2_path: P,
    ) -> Result<(), Sbv2CoreError>
    where
        P: Into<PathBuf>;

    fn load_from_sbv2file_path<P>(
        &mut self,
        model_ident: &str,
        sbv2_path: P,
    ) -> Result<(), Sbv2CoreError>
    where
        P: Into<PathBuf>;
}

impl TtsModelHolderFromPath for TtsModelHolder {
    fn load_from_path<P>(
        &mut self,
        model_ident: &str,
        style_vectors_path: P,
        vits2_path: P,
    ) -> Result<(), Sbv2CoreError>
    where
        P: Into<PathBuf>,
    {
        let style_vectors_path: PathBuf = style_vectors_path.into();
        let vits2_path: PathBuf = vits2_path.into();

        let style_vectors_bytes = std::fs::read(style_vectors_path)?;
        let vits2_bytes = std::fs::read(vits2_path)?;

        self.load(model_ident, style_vectors_bytes, vits2_bytes)
    }

    fn load_from_sbv2file_path<P>(
        &mut self,
        model_ident: &str,
        sbv2_path: P,
    ) -> Result<(), Sbv2CoreError>
    where
        P: Into<PathBuf>,
    {
        let sbv2_path: PathBuf = sbv2_path.into();
        let sbv2_bytes = std::fs::read(sbv2_path)?;

        self.load_from_sbv2file(model_ident, sbv2_bytes)
    }
}

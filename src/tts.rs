use std::{
    io::{Cursor, Read as _},
    path::PathBuf,
};

use ndarray::{Array1, Array2, Array3, Axis};
use ort::Session;
use tokenizers::Tokenizer;

use crate::{errors::Sbv2CoreError, jtalk::JTalk};

#[derive(Debug)]
struct NoUpperLimitTtsModel {
    model_ident: String,

    vits2: Session,
    style_vectors: Array2<f32>,
}

#[derive(Debug)]
struct UpperLimitTtsModel {
    model_ident: String,

    vits2: Option<Session>,
    bytes: Vec<u8>,
    style_vectors: Array2<f32>,
}

#[derive(Debug)]
enum EitherTtsModel<'a> {
    Limit(&'a UpperLimitTtsModel),
    NoLimit(&'a NoUpperLimitTtsModel),
}

#[derive(Debug)]
enum EitherTtsModelVec {
    Limit(Vec<UpperLimitTtsModel>),
    NoLimit(Vec<NoUpperLimitTtsModel>),
}

pub struct TtsModelHolder {
    models: EitherTtsModelVec,
    max_loaded_models: Option<usize>,

    bert: Session,
    tokenizer: Tokenizer,
    jtalk: JTalk,
}

impl TtsModelHolder {
    pub fn new<T>(
        bert_model_bytes: T,
        tokenizer_bytes: T,
        max_loaded_models: Option<usize>,
    ) -> Result<Self, Sbv2CoreError>
    where
        T: AsRef<[u8]>,
    {
        let bert = crate::model::load_model_session(bert_model_bytes, true)?;
        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)?;

        let models = match max_loaded_models {
            Some(_) => EitherTtsModelVec::Limit(vec![]),
            None => EitherTtsModelVec::NoLimit(vec![]),
        };

        Ok(TtsModelHolder {
            bert,
            tokenizer,
            jtalk: JTalk::new()?,
            models,
            max_loaded_models,
        })
    }

    pub fn new_from_filepath<P>(
        bert_model: P,
        tokenizer: P,
        max_loaded_models: Option<usize>,
    ) -> Result<Self, Sbv2CoreError>
    where
        P: Into<PathBuf>,
    {
        let bert_model: PathBuf = bert_model.into();
        let tokenizer: PathBuf = tokenizer.into();

        let bert_model_bytes = std::fs::read(bert_model)?;
        let tokenizer_bytes = std::fs::read(tokenizer)?;

        Self::new(bert_model_bytes, tokenizer_bytes, max_loaded_models)
    }

    pub fn get_loadedmodel_count(&self) -> usize {
        let models = match &self.models {
            EitherTtsModelVec::Limit(vec) => vec,
            EitherTtsModelVec::NoLimit(vec) => return vec.len(),
        };

        let mut loaded_model_count = 0;
        for i in models {
            if i.vits2.is_some() {
                loaded_model_count += 1;
            }
        }

        loaded_model_count
    }

    pub fn is_max_models_loaded(&self) -> bool {
        let models = match &self.models {
            EitherTtsModelVec::Limit(vec) => vec,
            EitherTtsModelVec::NoLimit(_) => return false,
        };

        let Some(upper_limit) = self.max_loaded_models else {
            return false;
        };

        let mut loaded_model_count = 0;
        for i in models {
            if i.vits2.is_some() {
                loaded_model_count += 1;
            }
        }

        loaded_model_count >= upper_limit
    }

    pub fn load<T>(
        &mut self,
        model_ident: &str,
        style_vectors_bytes: T,
        vits2_bytes: Vec<u8>,
    ) -> Result<(), Sbv2CoreError>
    where
        T: AsRef<[u8]>,
    {
        if self.get_either_model(model_ident).is_some() {
            return Ok(());
        };

        let max_loaded = self.is_max_models_loaded();

        match &mut self.models {
            EitherTtsModelVec::Limit(vec) => {
                let style_vectors = crate::style::load_style(style_vectors_bytes)?;
                let session = if max_loaded {
                    None
                } else {
                    Some(crate::model::load_model_session(&vits2_bytes, false)?)
                };

                let model = UpperLimitTtsModel {
                    model_ident: model_ident.to_string(),
                    vits2: session,
                    bytes: vits2_bytes,
                    style_vectors,
                };

                vec.push(model);
            }

            EitherTtsModelVec::NoLimit(vec) => {
                let style_vectors = crate::style::load_style(style_vectors_bytes)?;
                let session = crate::model::load_model_session(&vits2_bytes, false)?;

                let model = NoUpperLimitTtsModel {
                    model_ident: model_ident.to_string(),
                    vits2: session,
                    style_vectors,
                };
                vec.push(model);
            }
        }

        Ok(())
    }

    pub fn unload(&mut self, model_ident: &str) -> bool {
        let mut result = false;

        match &mut self.models {
            EitherTtsModelVec::Limit(vec) => {
                if let Some((i, _)) = vec
                    .iter()
                    .enumerate()
                    .find(|(_, m)| m.model_ident == model_ident)
                {
                    vec.remove(i);
                    result = true;
                };
            }

            EitherTtsModelVec::NoLimit(vec) => {
                if let Some((i, _)) = vec
                    .iter()
                    .enumerate()
                    .find(|(_, m)| m.model_ident == model_ident)
                {
                    vec.remove(i);
                    result = true;
                };
            }
        }

        result
    }

    pub fn load_from_sbv2file<T>(
        &mut self,
        model_ident: &str,
        sbv2file_bytes: T,
    ) -> Result<(), Sbv2CoreError>
    where
        T: AsRef<[u8]>,
    {
        // .sbv2 ファイルから vits2 ファイルと style_vectors.json ファイルを取得
        let (vits2_bytes, style_vectors_bytes) = {
            let decoded = zstd::decode_all(Cursor::new(sbv2file_bytes.as_ref()))?;
            let mut archive = tar::Archive::new(Cursor::new(decoded));
            let mut entries = archive.entries()?;

            let mut vits2 = None;
            let mut style_vectors = None;

            while let Some(Ok(mut e)) = entries.next() {
                let mut file_bytes = Vec::with_capacity(e.size() as usize);
                e.read_to_end(&mut file_bytes)?;

                let file_name = String::from_utf8_lossy(&e.path_bytes()).to_string();
                match file_name.as_str() {
                    "model.onnx" => vits2 = Some(file_bytes),
                    "style_vectors.json" => style_vectors = Some(file_bytes),
                    _ => continue,
                }
            }

            let create_err = |content: &str| {
                return Err(Sbv2CoreError::ModelNotFoundError(content.to_string()));
            };

            match (vits2, style_vectors) {
                (Some(vits2), Some(style_vectors)) => (vits2, style_vectors),
                (None, Some(_)) => create_err("vits2 not found")?,
                (Some(_), None) => create_err("style_vectors not found")?,
                (None, None) => create_err("vits2, style_vectors not found")?,
            }
        };

        self.load(model_ident, style_vectors_bytes, vits2_bytes)
    }

    pub fn model_idents(&self) -> Vec<String> {
        match &self.models {
            EitherTtsModelVec::Limit(vec) => {
                vec.iter().map(|m| m.model_ident.to_string()).collect()
            }
            EitherTtsModelVec::NoLimit(vec) => {
                vec.iter().map(|m| m.model_ident.to_string()).collect()
            }
        }
    }

    // 上限が設定されている場合モデルのsessionをNoneにする
    fn session_unload(&mut self, model_ident: &str) -> Result<(), Sbv2CoreError> {
        let models = match &mut self.models {
            EitherTtsModelVec::Limit(vec) => vec,
            EitherTtsModelVec::NoLimit(_) => return Ok(()),
        };

        let Some(model) = models.iter_mut().find(|i| i.model_ident == model_ident) else {
            return Ok(());
        };

        model.vits2 = None;

        Ok(())
    }

    // sessionの上限が設定されていてモデルのsessionが読み込まれていないならbytesから読み込む
    fn model_session_preparation(&mut self, model_ident: &str) -> Result<(), Sbv2CoreError> {
        let model = {
            let models = match &mut self.models {
                EitherTtsModelVec::Limit(vec) => vec,
                EitherTtsModelVec::NoLimit(_) => return Ok(()),
            };

            let (idx, model) = models
                .iter()
                .enumerate()
                .find(|(_, i)| i.model_ident == model_ident)
                .ok_or(Sbv2CoreError::ModelNotFoundError(model_ident.to_string()))?;

            if model.vits2.is_some() {
                return Ok(());
            }

            models.remove(idx)
        };

        // すでに max - 1 (上で取り除いた分を除く) 個のモデルのSessionがsomeなら、一番古いものを取り除く
        if let Some(max_loaded_models) = self.max_loaded_models {
            let loaded_model_count = self.get_loadedmodel_count();

            let remove_model_ident = {
                let models = match &mut self.models {
                    EitherTtsModelVec::Limit(vec) => vec,
                    EitherTtsModelVec::NoLimit(_) => return Ok(()),
                };

                if let Some(remove_model) = models.iter().find(|m| m.vits2.is_some()) {
                    remove_model.model_ident.clone()
                } else {
                    "".to_string()
                }
            };

            if loaded_model_count >= (max_loaded_models - 1) {
                self.session_unload(&remove_model_ident)?;
            }
        };

        let sbv2_session = crate::model::load_model_session(&model.bytes, false)?;

        let models = match &mut self.models {
            EitherTtsModelVec::Limit(vec) => vec,
            EitherTtsModelVec::NoLimit(_) => return Ok(()),
        };

        models.push(UpperLimitTtsModel {
            model_ident: model_ident.to_string(),
            vits2: Some(sbv2_session),
            bytes: model.bytes,
            style_vectors: model.style_vectors,
        });

        Ok(())
    }

    fn get_either_model(&self, model_ident: &str) -> Option<EitherTtsModel> {
        let mut model = None;

        match &self.models {
            EitherTtsModelVec::Limit(vec) => {
                if let Some(limit_model) = vec.iter().find(|i| i.model_ident == model_ident) {
                    model = Some(EitherTtsModel::Limit(limit_model));
                }
            }
            EitherTtsModelVec::NoLimit(vec) => {
                if let Some(no_limit_model) = vec.iter().find(|i| i.model_ident == model_ident) {
                    model = Some(EitherTtsModel::NoLimit(no_limit_model));
                }
            }
        }

        model
    }

    fn parse_text(
        &self,
        text: &str,
    ) -> Result<(Array2<f32>, Array1<i64>, Array1<i64>, Array1<i64>), Sbv2CoreError> {
        crate::tts_util::parse_text_blocking(
            text,
            &self.jtalk,
            &self.tokenizer,
            |token_ids, attention_masks| {
                crate::bert::predict(&self.bert, token_ids, attention_masks)
            },
        )
    }

    pub fn synthesize(
        &mut self,
        model_ident: &str,
        text: &str,
        style_id: i32,
        speaker_id: i64,
        options: SynthesizeOptions,
    ) -> Result<Vec<u8>, Sbv2CoreError> {
        self.model_session_preparation(model_ident)?;

        let either_ttsmodel = self
            .get_either_model(model_ident)
            .ok_or(Sbv2CoreError::ModelNotFoundError(model_ident.to_string()))?;

        let (vits2, style_vectors) = match either_ttsmodel {
            EitherTtsModel::Limit(upper_limit_tts_model) => {
                let vits2 = upper_limit_tts_model.vits2.as_ref().expect("vits2 is None");
                let style_vectors = &upper_limit_tts_model.style_vectors;

                (vits2, style_vectors)
            }

            EitherTtsModel::NoLimit(no_upper_limit_tts_model) => {
                let vits2 = &no_upper_limit_tts_model.vits2;
                let style_vectors = &no_upper_limit_tts_model.style_vectors;

                (vits2, style_vectors)
            }
        };

        let style_vector =
            crate::style::get_style_vector(style_vectors, style_id, options.style_weight)?;

        let audio_array = match options.split_sentences {
            true => {
                let texts: Vec<&str> = text.split('\n').collect();

                let mut audios = vec![];
                for (i, t) in texts.iter().enumerate() {
                    if t.is_empty() {
                        continue;
                    }

                    let (bert_ori, phones, tones, lang_ids) = self.parse_text(t)?;

                    let audio = crate::model::synthesize(
                        vits2,
                        bert_ori.to_owned(),
                        phones,
                        Array1::from_vec(vec![speaker_id]),
                        tones,
                        lang_ids,
                        style_vector.clone(),
                        options.sdp_ratio,
                        options.length_scale,
                        0.677,
                        0.8,
                    )?;

                    audios.push(audio);
                    if i != texts.len() - 1 {
                        audios.push(Array3::zeros((1, 1, 22050)));
                    }
                }

                ndarray::concatenate(
                    Axis(2),
                    &audios.iter().map(|x| x.view()).collect::<Vec<_>>(),
                )?
            }

            false => {
                let (bert_ori, phones, tones, lang_ids) = self.parse_text(text)?;
                crate::model::synthesize(
                    vits2,
                    bert_ori.to_owned(),
                    phones,
                    Array1::from_vec(vec![speaker_id]),
                    tones,
                    lang_ids,
                    style_vector,
                    options.sdp_ratio,
                    options.length_scale,
                    0.677,
                    0.8,
                )?
            }
        };

        crate::tts_util::array_to_vec(audio_array)
    }
}

/// Synthesize options
///
/// # Fields
/// - `sdp_ratio`: SDP ratio
/// - `length_scale`: Length scale
/// - `style_weight`: Style weight
/// - `split_sentences`: Split sentences
pub struct SynthesizeOptions {
    pub sdp_ratio: f32,
    pub length_scale: f32,
    pub style_weight: f32,
    pub split_sentences: bool,
}

impl Default for SynthesizeOptions {
    fn default() -> Self {
        SynthesizeOptions {
            sdp_ratio: 0.0,
            length_scale: 1.0,
            style_weight: 1.0,
            split_sentences: true,
        }
    }
}

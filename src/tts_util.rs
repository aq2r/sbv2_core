use std::io::Cursor;

use hound::{SampleFormat, WavSpec, WavWriter};
use ndarray::{s, Array, Array1, Array2, Array3, Axis};
use tokenizers::Tokenizer;

use crate::{errors::Sbv2CoreError, jtalk::JTalk};

pub fn parse_text_blocking(
    text: &str,
    jtalk: &JTalk,
    tokenizer: &Tokenizer,
    bert_predict: impl FnOnce(Vec<i64>, Vec<i64>) -> Result<ndarray::Array2<f32>, Sbv2CoreError>,
) -> Result<(Array2<f32>, Array1<i64>, Array1<i64>, Array1<i64>), Sbv2CoreError> {
    let text = jtalk.num2word(text)?;
    let normalized_text = crate::norm::normalize_text(&text);

    let process = jtalk.process_text(&normalized_text)?;
    let (phones, tones, mut word2ph) = process.g2p()?;
    let (phones, tones, lang_ids) = crate::nlp::cleaned_text_to_sequence(phones, tones);

    let phones = crate::utils::intersperse(&phones, 0);
    let tones = crate::utils::intersperse(&tones, 0);
    let lang_ids = crate::utils::intersperse(&lang_ids, 0);

    for item in &mut word2ph {
        *item *= 2;
    }
    word2ph[0] += 1;

    let text = {
        let (seq_text, _) = process.text_to_seq_kata()?;
        seq_text.join("")
    };
    let (token_ids, attention_masks) = crate::tokenizer::tokenize(&text, tokenizer)?;

    assert!(
        word2ph.len() == text.chars().count() + 2,
        "{} {}",
        word2ph.len(),
        normalized_text.chars().count()
    );

    let bert_content = bert_predict(token_ids, attention_masks)?;

    let mut phone_level_feature = vec![];
    for (i, reps) in word2ph.iter().enumerate() {
        let repeat_feature = {
            let (reps_rows, reps_cols) = (*reps, 1);
            let arr_len = bert_content.slice(s![i, ..]).len();

            let mut results: Array2<f32> = Array::zeros((reps_rows as usize, arr_len * reps_cols));

            for j in 0..reps_rows {
                for k in 0..reps_cols {
                    let mut view = results.slice_mut(s![j, k * arr_len..(k + 1) * arr_len]);
                    view.assign(&bert_content.slice(s![i, ..]));
                }
            }
            results
        };
        phone_level_feature.push(repeat_feature);
    }

    let phone_level_feature = ndarray::concatenate(
        Axis(0),
        &phone_level_feature
            .iter()
            .map(|x| x.view())
            .collect::<Vec<_>>(),
    )?;

    let bert_ori = phone_level_feature.t();
    Ok((
        bert_ori.to_owned(),
        phones.into(),
        tones.into(),
        lang_ids.into(),
    ))
}

pub fn array_to_vec(audio_array: Array3<f32>) -> Result<Vec<u8>, Sbv2CoreError> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut cursor = Cursor::new(Vec::new());

    let mut writer = WavWriter::new(&mut cursor, spec)?;
    for i in 0..audio_array.shape()[0] {
        let output = audio_array.slice(s![i, 0, ..]).to_vec();
        for sample in output {
            writer.write_sample(sample)?;
        }
    }
    writer.finalize()?;

    Ok(cursor.into_inner())
}

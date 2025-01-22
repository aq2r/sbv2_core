use std::{
    cmp::Reverse,
    collections::HashSet,
    sync::{Arc, LazyLock},
};

use jpreprocess::{
    error::JPreprocessError, kind::JPreprocessDictionaryKind, DefaultFetcher, JPreprocess,
    JPreprocessConfig, SystemDictionaryConfig,
};
use regex::Regex;

use crate::{
    errors::Sbv2CoreError,
    mora::{MORA_KATA_TO_MORA_PHONEMES, VOWELS},
    norm::PUNCTUATIONS,
};

type JPreprocessType = JPreprocess<DefaultFetcher>;

pub(crate) struct JTalk {
    pub jpreprocess: Arc<JPreprocessType>,
}

impl JTalk {
    pub fn new() -> Result<Self, JPreprocessError> {
        let initialized = {
            let config = JPreprocessConfig {
                dictionary: SystemDictionaryConfig::Bundled(JPreprocessDictionaryKind::NaistJdic),
                user_dictionary: None,
            };
            JPreprocess::from_config(config)?
        };

        Ok(JTalk {
            jpreprocess: Arc::new(initialized),
        })
    }

    pub fn num2word(&self, text: &str) -> Result<String, JPreprocessError> {
        let mut parsed = self.jpreprocess.text_to_njd(text)?;
        parsed.preprocess();

        let texts: Vec<String> = parsed
            .nodes
            .iter()
            .map(|x| x.get_string().to_string())
            .collect();

        // テキストが空白だった時のため?
        Ok(texts.join(""))
    }

    pub fn process_text(&self, text: &str) -> Result<JTalkProcess, JPreprocessError> {
        let parsed = self.jpreprocess.run_frontend(text)?;
        let jtalk_process = JTalkProcess::new(Arc::clone(&self.jpreprocess), parsed);

        Ok(jtalk_process)
    }
}

macro_rules! hash_set {
    ($($elem:expr),* $(,)?) => {{
        let mut set = HashSet::new();
        $(
            set.insert($elem);
        )*
        set
    }};
}

pub struct JTalkProcess {
    jpreprocess: Arc<JPreprocessType>,
    parsed: Vec<String>,
}

impl JTalkProcess {
    fn new(jpreprocess: Arc<JPreprocessType>, parsed: Vec<String>) -> Self {
        Self {
            jpreprocess,
            parsed,
        }
    }

    fn fix_phone_tone(
        &self,
        phone_tone_list: Vec<(String, i32)>,
    ) -> Result<Vec<(String, i32)>, Sbv2CoreError> {
        let tone_values: HashSet<i32> = phone_tone_list.iter().map(|(_, tone)| *tone).collect();

        match tone_values.len() {
            1 => {
                assert!(tone_values == hash_set![0], "{:?}", tone_values);
                Ok(phone_tone_list)
            }

            2 => {
                if tone_values == hash_set![0, 1] {
                    Ok(phone_tone_list)
                } else if tone_values == hash_set![-1, 0] {
                    let fixed = phone_tone_list
                        .into_iter()
                        .map(|x| {
                            let new_tone = if x.1 == -1 { 0 } else { 1 };
                            (x.0, new_tone)
                        })
                        .collect();

                    Ok(fixed)
                } else {
                    Err(Sbv2CoreError::ValueError(
                        "Invalid tone values 0".to_string(),
                    ))
                }
            }

            _ => Err(Sbv2CoreError::ValueError(
                "Invalid tone values 1".to_string(),
            )),
        }
    }

    pub fn g2p(&self) -> Result<(Vec<String>, Vec<i32>, Vec<i32>), Sbv2CoreError> {
        let phone_tone_list_wo_punct = self.g2phone_tone_wo_punct()?;
        let (seq_text, seq_kata) = self.text_to_seq_kata()?;

        let sep_phonemes = JTalkProcess::handle_long(
            seq_kata
                .iter()
                .map(|x| JTalkProcess::kata_to_phoneme_list(x.clone()).unwrap())
                .collect(),
        );

        let phone_w_punct: Vec<String> = sep_phonemes
            .iter()
            .flat_map(|x| x.iter())
            .cloned()
            .collect();

        let mut phone_tone_list =
            JTalkProcess::align_tones(phone_w_punct, phone_tone_list_wo_punct)?;

        let mut sep_tokenized: Vec<Vec<String>> = Vec::new();
        for seq_text_item in &seq_text {
            let text = seq_text_item.as_str();

            if !PUNCTUATIONS.contains(&text) {
                sep_tokenized.push(text.chars().map(|x| x.to_string()).collect());
            } else {
                sep_tokenized.push(vec![text.to_string()]);
            }
        }

        let mut word2ph = Vec::new();
        for (token, phoneme) in sep_tokenized.iter().zip(sep_phonemes.iter()) {
            let phone_len = phoneme.len() as i32;
            let word_len = token.len() as i32;

            word2ph.append(&mut JTalkProcess::distribute_phone(phone_len, word_len));
        }

        let mut new_phone_tone_list = vec![("_".to_string(), 0)];
        new_phone_tone_list.append(&mut phone_tone_list);
        new_phone_tone_list.push(("_".to_string(), 0));

        let mut new_word2ph = vec![1];
        new_word2ph.extend(word2ph);
        new_word2ph.push(1);

        let phones: Vec<String> = new_phone_tone_list.iter().map(|(x, _)| x.clone()).collect();
        let tones: Vec<i32> = new_phone_tone_list.iter().map(|(_, x)| *x).collect();

        Ok((phones, tones, new_word2ph))
    }

    fn distribute_phone(n_phone: i32, n_word: i32) -> Vec<i32> {
        let mut phones_per_word = vec![0; n_word as usize];

        for _ in 0..n_phone {
            let min_task = phones_per_word.iter().min().unwrap();
            let min_index = phones_per_word
                .iter()
                .position(|&x| x == *min_task)
                .unwrap();

            phones_per_word[min_index] += 1;
        }
        phones_per_word
    }

    fn align_tones(
        phone_with_punct: Vec<String>,
        phone_tone_list: Vec<(String, i32)>,
    ) -> Result<Vec<(String, i32)>, Sbv2CoreError> {
        let mut result: Vec<(String, i32)> = Vec::new();

        let mut tone_index = 0;
        for phone in phone_with_punct.clone() {
            if tone_index >= phone_tone_list.len() {
                result.push((phone, 0));
            } else if phone == phone_tone_list[tone_index].0 {
                result.push((phone, phone_tone_list[tone_index].1));
                tone_index += 1;
            } else if PUNCTUATIONS.contains(&phone.as_str()) {
                result.push((phone, 0));
            } else {
                println!("phones {:?}", phone_with_punct);
                println!("phone_tone_list: {:?}", phone_tone_list);
                println!("result: {:?}", result);
                println!("tone_index: {:?}", tone_index);
                println!("phone: {:?}", phone);

                return Err(Sbv2CoreError::ValueError(format!(
                    "Mismatched phoneme: {}",
                    phone
                )));
            }
        }

        Ok(result)
    }

    fn handle_long(mut sep_phonemes: Vec<Vec<String>>) -> Vec<Vec<String>> {
        for i in 0..sep_phonemes.len() {
            if sep_phonemes[i].is_empty() {
                continue;
            }

            if sep_phonemes[i][0] == "ー" {
                match i {
                    0 => sep_phonemes[i][0] = "ー".to_string(),

                    _ => {
                        let prev_phoneme = sep_phonemes[i - 1].last().unwrap();

                        if VOWELS.contains(&prev_phoneme.as_str()) {
                            sep_phonemes[i][0] = prev_phoneme.clone();
                        } else {
                            sep_phonemes[i][0] = "ー".to_string();
                        }
                    }
                }
            }

            if sep_phonemes[i].iter().any(|x| x.as_str() == "ー") {
                for e in 0..sep_phonemes[i].len() {
                    if sep_phonemes[i][e] == "ー" {
                        sep_phonemes[i][e] =
                            sep_phonemes[i][e - 1].chars().last().unwrap().to_string();
                    }
                }
            }
        }

        sep_phonemes
    }

    fn kata_to_phoneme_list(mut text: String) -> Result<Vec<String>, Sbv2CoreError> {
        let chars: HashSet<String> = text.chars().map(|x| x.to_string()).collect();

        static SUBSET_HASHSET: LazyLock<HashSet<String>> =
            LazyLock::new(|| HashSet::from_iter(PUNCTUATIONS.iter().map(|x| x.to_string())));

        if chars.is_subset(&SUBSET_HASHSET) {
            return Ok(text.chars().map(|x| x.to_string()).collect());
        }

        static KATAKANA_PATTERN: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"[\u30A0-\u30FF]+").unwrap());

        if !KATAKANA_PATTERN.is_match(&text) {
            return Err(Sbv2CoreError::ValueError(format!(
                "Input must be katakana only: {}",
                text
            )));
        }

        static MORA_PATTERN: LazyLock<Vec<String>> = LazyLock::new(|| {
            let mut sorted_keys: Vec<String> = MORA_KATA_TO_MORA_PHONEMES.keys().cloned().collect();
            sorted_keys.sort_by_key(|b| Reverse(b.len()));

            sorted_keys
        });

        for mora in MORA_PATTERN.iter() {
            let (consonant, vowel) = &MORA_KATA_TO_MORA_PHONEMES[mora];

            if consonant.is_none() {
                text = text.replace(mora, &format!(" {}", vowel));
            } else {
                text = text.replace(mora, &format!(" {} {}", consonant.as_ref().unwrap(), vowel));
            }
        }

        let longvowel_replacement = |m: &regex::Captures| {
            let capture_1 = &m[1];
            let capture_2 = &m[2];

            let mut add_string = String::new();
            for _ in 0..capture_2.chars().count() {
                add_string += " ";
                add_string += capture_1;
            }

            format!("{capture_1}{add_string}")
        };

        static LONG_PATTERN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\w)(ー*)").unwrap());

        text = LONG_PATTERN
            .replace_all(&text, longvowel_replacement)
            .to_string();

        let data: Vec<_> = text.trim().split(' ').map(|x| x.to_string()).collect();

        Ok(data)
    }

    pub fn text_to_seq_kata(&self) -> Result<(Vec<String>, Vec<String>), Sbv2CoreError> {
        let mut seq_kata = vec![];
        let mut seq_text = vec![];

        for parts in &self.parsed {
            let part_lists: Vec<String> = parts.split(',').map(|x| x.to_string()).collect();
            let (string, pron) = (&part_lists[0], &part_lists[9]);

            let mut yomi = pron.replace('’', "");

            let word = crate::norm::replace_punctuation(string.clone());
            assert!(!yomi.is_empty(), "Empty yomi: {}", word);

            match yomi.as_str() {
                "、" => {
                    if !word
                        .chars()
                        .all(|x| PUNCTUATIONS.contains(&x.to_string().as_str()))
                    {
                        yomi = "'".repeat(word.len());
                    } else {
                        yomi = word.clone();
                    }
                }

                "？" => {
                    assert!(word == "?", "yomi `？` comes from: {}", word);
                    yomi = "?".to_string();
                }

                _ => (),
            }

            seq_text.push(word);
            seq_kata.push(yomi);
        }

        Ok((seq_text, seq_kata))
    }

    fn g2phone_tone_wo_punct(&self) -> Result<Vec<(String, i32)>, Sbv2CoreError> {
        let prosodies = self.g2p_prosody()?;

        let mut results: Vec<(String, i32)> = Vec::new();
        let mut current_phrase: Vec<(String, i32)> = Vec::new();
        let mut current_tone = 0;

        for (i, letter) in prosodies.iter().enumerate() {
            match letter.as_str() {
                "^" => {
                    assert!(i == 0);
                }

                "$" | "?" | "_" | "#" => {
                    results.extend(self.fix_phone_tone(current_phrase.clone())?);

                    if let "$" | "?" = letter.as_str() {
                        assert!(i == prosodies.len() - 1);
                    }

                    current_phrase = Vec::new();
                    current_tone = 0;
                }

                "[" => current_tone += 1,
                "]" => current_tone -= 1,

                _ => {
                    let new_letter = if letter == "cl" {
                        "q".to_string()
                    } else {
                        letter.clone()
                    };

                    current_phrase.push((new_letter, current_tone));
                }
            }
        }

        Ok(results)
    }

    fn g2p_prosody(&self) -> Result<Vec<String>, Sbv2CoreError> {
        let labels = self.jpreprocess.make_label(self.parsed.clone());

        let mut phones: Vec<String> = Vec::new();
        for (i, label) in labels.iter().enumerate() {
            let mut p3 = label.phoneme.c.clone().unwrap();
            if "AIUEO".contains(&p3) {
                // 文字をlowerする
                p3 = p3.to_lowercase();
            }

            match p3.as_str() {
                "sil" => {
                    assert!(i == 0 || i == labels.len() - 1);

                    if i == 0 {
                        phones.push("^".to_string());
                    } else if i == labels.len() - 1 {
                        let e3 = label.accent_phrase_prev.as_ref().unwrap().is_interrogative;
                        match e3 {
                            true => phones.push("$".to_string()),
                            false => phones.push("?".to_string()),
                        }
                    }

                    continue;
                }

                "pau" => {
                    phones.push("_".to_string());
                    continue;
                }

                _ => {
                    phones.push(p3.clone());
                }
            }

            let a1 = if let Some(mora) = &label.mora {
                mora.relative_accent_position as i32
            } else {
                -50
            };
            let a2 = if let Some(mora) = &label.mora {
                mora.position_forward as i32
            } else {
                -50
            };
            let a3 = if let Some(mora) = &label.mora {
                mora.position_backward as i32
            } else {
                -50
            };

            let f1 = if let Some(accent_phrase) = &label.accent_phrase_curr {
                accent_phrase.mora_count as i32
            } else {
                -50
            };

            let a2_next = if let Some(mora) = &labels[i + 1].mora {
                mora.position_forward as i32
            } else {
                -50
            };

            if a3 == 1 && a2_next == 1 && "aeiouAEIOUNcl".contains(&p3) {
                phones.push("#".to_string());
            } else if a1 == 0 && a2_next == a2 + 1 && a2 != f1 {
                phones.push("]".to_string());
            } else if a2 == 1 && a2_next == 2 {
                phones.push("[".to_string());
            }
        }

        Ok(phones)
    }
}

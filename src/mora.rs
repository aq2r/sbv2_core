use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::LazyLock};

#[derive(Debug, Serialize, Deserialize)]
pub struct Mora {
    pub mora: String,
    pub consonant: Option<String>,
    pub vowel: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MoraFile {
    pub minimum: Vec<Mora>,
    pub additional: Vec<Mora>,
}

static MORA_LIST_MINIMUM: LazyLock<Vec<Mora>> = LazyLock::new(|| {
    let data: MoraFile = serde_json::from_str(include_str!("./mora_list.json")).unwrap();
    data.minimum
});

static MORA_LIST_ADDITIONAL: LazyLock<Vec<Mora>> = LazyLock::new(|| {
    let data: MoraFile = serde_json::from_str(include_str!("./mora_list.json")).unwrap();
    data.additional
});

pub static MORA_KATA_TO_MORA_PHONEMES: LazyLock<HashMap<String, (Option<String>, String)>> =
    LazyLock::new(|| {
        let mut map = HashMap::new();

        for mora in MORA_LIST_MINIMUM.iter().chain(MORA_LIST_ADDITIONAL.iter()) {
            map.insert(
                mora.mora.clone(),
                (mora.consonant.clone(), mora.vowel.clone()),
            );
        }

        map
    });

pub const VOWELS: [&str; 6] = ["a", "i", "u", "e", "o", "N"];

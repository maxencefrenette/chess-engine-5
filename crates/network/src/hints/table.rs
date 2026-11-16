use std::collections::{BTreeMap, HashMap};

use candle_core::{Device, Result, Tensor};

use super::{FEATURES, HintConfig};

pub type MaterialKey = [u8; 10];

pub fn material_key(board: &[u8; 32], black: bool) -> MaterialKey {
    let mut key = [0; 10];
    for byte in board {
        for piece in [byte & 15, byte >> 4] {
            if piece == 0 || piece == 6 || piece == 12 || piece > 12 {
                continue;
            }
            let side = usize::from(piece > 6) ^ usize::from(black);
            key[side * 5 + usize::from((piece - 1) % 6)] += 1;
        }
    }
    key
}

type GameOutcomes = BTreeMap<(usize, u64), [Option<i8>; 2]>;

// A game has one outcome per side. Repeated rows of either side do not add weight.
#[derive(Default)]
pub struct TableBuilder {
    games: BTreeMap<MaterialKey, GameOutcomes>,
}

impl TableBuilder {
    pub fn observe(&mut self, dataset: usize, game: u64, key: MaterialKey, black: bool, value: i8) {
        self.games
            .entry(key)
            .or_default()
            .entry((dataset, game))
            .or_insert([None; 2])[usize::from(black)] = Some(value);
    }

    pub fn finish(self) -> MaterialTable {
        let entries = self
            .games
            .into_iter()
            .map(|(key, games)| {
                let mut entry = [0.; FEATURES];
                entry[3] = games.len() as f32;
                for sides in games.into_values() {
                    let count = sides.iter().flatten().count() as f32;
                    for value in sides.into_iter().flatten() {
                        entry[(value + 1) as usize] += 1. / count;
                    }
                }
                (key, entry)
            })
            .collect();
        MaterialTable { entries }
    }
}

#[derive(Clone, Default)]
pub struct MaterialTable {
    entries: HashMap<MaterialKey, [f32; FEATURES]>,
}

impl MaterialTable {
    pub fn lookup(&self, key: &MaterialKey, config: HintConfig) -> [f32; FEATURES] {
        let HintConfig::Material { min_games, .. } = config else {
            return [0.; FEATURES];
        };
        let Some(entry) = self.entries.get(key) else {
            return [0.; FEATURES];
        };
        let games = entry[3];
        if games < min_games.get() as f32 {
            return [0.; FEATURES];
        }
        // One pseudo-count per outcome avoids certainty from a small sample.
        let p = [0, 1, 2].map(|i| (entry[i] + 1.) / (games + 3.));
        let support = ((games / min_games.get() as f32).log2().floor() as usize).min(7);
        let concentration = ((p.into_iter().fold(0., f32::max) - 1. / 3.) * 1.5 * 4.) as usize;
        [
            p[0],
            p[1],
            p[2],
            (1 + support * 4 + concentration.min(3)) as f32,
        ]
    }

    pub fn save(&self, tensors: &mut HashMap<String, Tensor>) -> Result<()> {
        let mut entries: Vec<_> = self.entries.iter().collect();
        entries.sort_by_key(|(key, _)| **key);
        let size = entries.len();
        let keys = entries
            .iter()
            .flat_map(|(k, _)| k.iter().copied())
            .collect::<Vec<_>>();
        let values = entries
            .iter()
            .flat_map(|(_, v)| v.iter().copied())
            .collect::<Vec<_>>();
        tensors.insert(
            "material_table.keys".into(),
            Tensor::from_vec(keys, (size, 10), &Device::Cpu)?,
        );
        tensors.insert(
            "material_table.values".into(),
            Tensor::from_vec(values, (size, FEATURES), &Device::Cpu)?,
        );
        Ok(())
    }

    pub fn load(tensors: &mut HashMap<String, Tensor>) -> Result<Self> {
        let keys = tensors
            .remove("material_table.keys")
            .ok_or_else(|| candle_core::Error::Msg("missing material table keys".into()))?;
        let values = tensors
            .remove("material_table.values")
            .ok_or_else(|| candle_core::Error::Msg("missing material table values".into()))?;
        let size = keys.dim(0)?;
        if keys.dims() != [size, 10] || values.dims() != [size, FEATURES] {
            candle_core::bail!("invalid material table shape");
        }
        let keys = keys.to_vec2::<u8>()?;
        let values = values.to_vec2::<f32>()?;
        let mut entries = HashMap::new();
        for (key, value) in keys.into_iter().zip(values) {
            let key: MaterialKey = key.try_into().expect("validated key shape");
            let value: [f32; FEATURES] = value.try_into().expect("validated value shape");
            if value.iter().any(|v| !v.is_finite() || *v < 0.)
                || value[3] < 1.
                || (value[..3].iter().sum::<f32>() - value[3]).abs() > 0.01
                || entries.insert(key, value).is_some()
            {
                candle_core::bail!("invalid material table entry");
            }
        }
        Ok(Self { entries })
    }
}

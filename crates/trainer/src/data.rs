use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use candle_core::{Device, Tensor};
pub use chess_data::Result;
use chess_data::{discover_shards, read_shard};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

use chess_network::{INPUTS, SQUARES, encode_position, policy_inputs::PolicyInputs};

pub struct Row {
    features: Vec<f32>,
    moves: Vec<(u16, u16)>,
    reply: Option<crate::reply::Reply>,
    total_visits: f32,
    pub value: [f32; 3],
    use_value: bool,
}

pub struct Dataset {
    pub train: Vec<Row>,
    pub test: Vec<Row>,
    pub target_seconds: f64,
    pub value_labels_per_game: Vec<usize>,
    pub test_ids: Vec<u64>,
    pub estimated_generation_seconds: f64,
}

pub fn curriculum_shards<'a>(
    datasets: impl IntoIterator<Item = &'a Path>,
) -> Result<Vec<Vec<PathBuf>>> {
    let mut seen = BTreeSet::new();
    let mut phases = Vec::new();
    for dataset in datasets {
        let shards = discover_shards(dataset)?;
        for shard in &shards {
            if !seen.insert(fs::canonicalize(shard)?) {
                return Err(
                    format!("curriculum uses shard {} more than once", shard.display()).into(),
                );
            }
        }
        phases.push(shards);
    }
    Ok(phases)
}

pub fn load(
    shards: &[PathBuf],
    config: &crate::config::Config,
    train_positions: usize,
    retain_holdout: bool,
) -> Result<Dataset> {
    let seed = config.seed;
    let value_sample_rate = config.value_sample_rate;
    let root_fraction = config.root_wdl_fraction as f32;
    let mut rows = Vec::new();
    for shard in shards {
        read_shard(shard, |row| {
            rows.push(row);
            Ok(())
        })?;
    }

    let mut games: Vec<_> = rows
        .iter()
        .map(|row| row.game_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if games.len() < 2 {
        return Err("at least two games are needed for a game-level split".into());
    }
    games.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
    let test_games = (games.len() / 10).max(1);
    let mut test_ids = games[..test_games].to_vec();
    test_ids.sort_unstable();
    let (test, mut train): (Vec<_>, Vec<_>) = rows
        .into_iter()
        .partition(|row| test_ids.binary_search(&row.game_id).is_ok());
    if train_positions < train.len() {
        train.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
        train.truncate(train_positions);
    }
    let mut by_game = BTreeMap::<_, Vec<_>>::new();
    for (index, row) in train.iter().enumerate() {
        by_game.entry(row.game_id).or_default().push(index);
    }
    let mut mask = vec![false; train.len()];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut value_labels_per_game = Vec::with_capacity(by_game.len());
    for (&game, indices) in &mut by_game {
        let count = ((indices.len() as f64 * value_sample_rate).round() as usize).max(1);
        if count < indices.len() {
            rng.set_stream(game);
            rng.set_word_pos(0);
            indices.shuffle(&mut rng);
        }
        for &index in &indices[..count] {
            mask[index] = true;
        }
        value_labels_per_game.push(count);
    }
    let estimated_generation_seconds = train
        .iter()
        .map(|row| u128::from(row.generation_ns_per_position))
        .sum::<u128>() as f64
        / 1e9;
    let started = std::time::Instant::now();
    let replies = if config.reply_weight > 0. {
        crate::reply::targets(&train, config.model.perspective)
    } else {
        (0..train.len()).map(|_| None).collect()
    };
    let target_seconds = started.elapsed().as_secs_f64();
    Ok(Dataset {
        target_seconds,
        train: train
            .into_iter()
            .zip(mask)
            .zip(replies)
            .map(|((row, use_value), reply)| {
                let mut encoded =
                    encode_row(row, root_fraction, use_value, config.model.perspective)?;
                encoded.reply = reply;
                Ok(encoded)
            })
            .collect::<Result<_>>()?,
        test: test
            .into_iter()
            .filter(|_| retain_holdout)
            .map(|row| encode_row(row, root_fraction, true, config.model.perspective))
            .collect::<Result<_>>()?,
        value_labels_per_game,
        test_ids,
        estimated_generation_seconds,
    })
}

fn encode_row(
    row: chess_data::Row,
    root_fraction: f32,
    use_value: bool,
    perspective: chess_network::perspective::Perspective,
) -> Result<Row> {
    let root = row.position.root_wdl();
    let mut outcome = [0.; 3];
    outcome[(row.value + 1) as usize] = 1.;
    let mut moves = row.position.policy;
    let black_to_move = !row.position.ply.is_multiple_of(2);
    let (board, black, mask) = perspective.orient(row.position.snapshot, black_to_move);
    let mut features = Vec::with_capacity(SQUARES * INPUTS);
    if mask != 0 {
        for (mv, _) in &mut moves {
            *mv ^= mask;
        }
    }
    encode_position(&mut features, &board, black).ok_or("invalid piece code")?;
    Ok(Row {
        features,
        reply: None,
        total_visits: moves.iter().map(|&(_, visits)| f32::from(visits)).sum(),
        moves,
        value: std::array::from_fn(|i| (1. - root_fraction) * outcome[i] + root_fraction * root[i]),
        use_value,
    })
}

pub struct Batch {
    pub inputs: PolicyInputs,
    pub mask: Tensor,
    pub policy: Tensor,
    pub value: Option<Tensor>,
    pub reply: Option<crate::reply::Targets>,
    pub value_indices: Option<Tensor>,
    pub value_count: usize,
    pub uniform_loss: f64,
}

impl Batch {
    pub fn new<'a>(
        rows: impl ExactSizeIterator<Item = &'a Row> + Clone,
        device: &Device,
    ) -> candle_core::Result<Self> {
        let size = rows.len();
        let mut positions = Vec::with_capacity(size * SQUARES * INPUTS);
        for row in rows.clone() {
            positions.extend_from_slice(&row.features);
        }
        let moves = rows.clone().map(|row| row.moves.iter().map(|&(mv, _)| mv));
        let inputs = PolicyInputs::new(positions, moves, device)?;
        let width = inputs.move_width();
        let mut mask = vec![-1e9_f32; size * width];
        let mut policy = vec![0f32; size * width];
        let mut value = Vec::with_capacity(size);
        let mut value_indices = Vec::new();
        let mut uniform_loss = 0.;
        for (i, row) in rows.clone().enumerate() {
            if row.use_value {
                value.extend_from_slice(&row.value);
                value_indices.push(i as u32);
            }
            uniform_loss += (row.moves.len() as f64).ln();
            for (j, &(_, visits)) in row.moves.iter().enumerate() {
                let index = i * width + j;
                mask[index] = 0.;
                policy[index] = f32::from(visits) / row.total_visits;
            }
        }
        let shape = (size, width);
        let value_count = value_indices.len();
        Ok(Self {
            inputs,
            mask: Tensor::from_vec(mask, shape, device)?,
            policy: Tensor::from_vec(policy, shape, device)?,
            value: (value_count > 0)
                .then(|| Tensor::from_vec(value, (value_count, 3), device))
                .transpose()?,
            value_indices: if (1..size).contains(&value_count) {
                Some(Tensor::from_vec(value_indices, value_count, device)?)
            } else {
                None
            },
            reply: crate::reply::Targets::batch(rows.map(|row| row.reply.as_ref()), device)?,
            value_count,
            uniform_loss,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curriculum_rejects_a_shard_selected_by_both_directory_and_file() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let shard = dir.path().join("one.parquet");
        fs::File::create(&shard)?;
        let other = tempfile::NamedTempFile::new()?;
        assert_eq!(
            curriculum_shards([dir.path(), other.path()])?,
            [vec![shard.clone()], vec![other.path().to_owned()]]
        );
        assert!(curriculum_shards([dir.path(), &shard]).is_err());
        Ok(())
    }

    #[test]
    fn both_splits_mix_root_wdl_and_only_training_masks_values() -> Result<()> {
        use chess_config::TomlConfig;
        use chess_data::{Board, Position};
        let dir = tempfile::tempdir()?;
        let positions = [0, 1, 2, 3].map(|ply| Position {
            ply,
            snapshot: Board {
                board: [0; 32],
                castling: 0,
                ep_square: 64,
                halfmove_clock: 0,
                repetitions: 0,
            },
            policy: vec![(0, 1)],
            played_move: 0,
            root_q: 0.25,
            root_d: 0.5,
        });
        chess_data::write_shard(
            dir.path(),
            0,
            (0..4).map(|id| (id, positions.as_slice(), 0)),
            42,
        )?;
        let mut config = crate::config::Config::from_repo_file("training.toml")?;
        config.value_sample_rate = 0.5;
        config.reply_weight = 0.;
        let outcome = [0., 1., 0.];
        let shards = discover_shards(dir.path())?;
        for (fraction, target) in [
            (0.0, outcome),
            (0.25, [0.03125, 0.875, 0.09375]),
            (0.5, [0.0625, 0.75, 0.1875]),
            (1.0, [0.125, 0.5, 0.375]),
        ] {
            config.root_wdl_fraction = fraction;
            let data = load(&shards, &config, 12, true)?;
            assert_eq!(data.value_labels_per_game, [2, 2, 2]);
            assert_eq!(data.train.iter().filter(|row| row.use_value).count(), 6);
            assert_eq!(data.test.len(), 4);
            assert!(
                data.test
                    .iter()
                    .all(|row| row.use_value && row.value == target)
            );
            assert!(data.train.iter().all(|row| row.value == target));
        }
        Ok(())
    }
}

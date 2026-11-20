use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use candle_core::{Device, Tensor};
pub use chess_data::Result;
use chess_data::{discover_shards, read_shard};
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

use chess_network::{INPUTS, SQUARES, encode_position, policy_inputs::PolicyInputs};

pub struct Row {
    game: u64,
    black: bool,
    material: chess_network::hints::MaterialKey,
    outcome: i8,
    hint: [f32; chess_network::hints::FEATURES],
    features: Vec<f32>,
    moves: Vec<(u16, f32)>,
    reply: Option<crate::reply::Reply>,
    total_visits: f32,
    pub value: [f32; 3],
    use_value: bool,
}

pub struct Dataset {
    pub train: Vec<Row>,
    pub test: Vec<Row>,
    pub target_seconds: f64,
    pub train_games: usize,
    pub value_labels_per_game: Vec<usize>,
    pub test_ids: Vec<u64>,
    pub estimated_generation_seconds: f64,
}

pub fn check_disjoint<'a>(datasets: impl IntoIterator<Item = &'a Path>) -> Result<()> {
    let mut seen = BTreeSet::new();
    for dataset in datasets {
        for shard in discover_shards(dataset)? {
            if !seen.insert(fs::canonicalize(&shard)?) {
                return Err(
                    format!("curriculum uses shard {} more than once", shard.display()).into(),
                );
            }
        }
    }
    Ok(())
}

pub fn load(
    dataset: &Path,
    config: &crate::config::Config,
    train_positions: usize,
    retain_holdout: bool,
    rescorer: &mut crate::rescoring::Rescorer,
) -> Result<Dataset> {
    let seed = config.seed;
    let value_sample_rate = config.value_sample_rate;
    let root_fraction = config.root_wdl_fraction as f32;
    let mut rows = Vec::new();
    for shard in discover_shards(dataset)? {
        read_shard(&shard, |row| {
            rows.push(row);
            Ok(())
        })?;
    }

    let history = crate::rescoring::History::new(&rows);
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
        crate::reply::targets(&train)
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
                let outcome_wdl = if use_value && root_fraction < 1.0 {
                    let repeated = history.has_repeated(&row);
                    rescorer.value(&row, repeated)?
                } else {
                    crate::rescoring::game_value(row.value)
                };
                let policy = rescorer.policy(&row)?;
                let mut encoded = encode_row(&row, policy, outcome_wdl, root_fraction, use_value)?;
                encoded.reply = reply;
                Ok(encoded)
            })
            .collect::<Result<_>>()?,
        test: test
            .into_iter()
            .filter(|_| retain_holdout)
            .map(|row| {
                encode_row(
                    &row,
                    crate::rescoring::search_policy(&row),
                    crate::rescoring::game_value(row.value),
                    root_fraction,
                    true,
                )
            })
            .collect::<Result<_>>()?,
        train_games: by_game.len(),
        value_labels_per_game,
        test_ids,
        estimated_generation_seconds,
    })
}

fn encode_row(
    row: &chess_data::Row,
    moves: Vec<(u16, f32)>,
    outcome: [f32; 3],
    root_fraction: f32,
    use_value: bool,
) -> Result<Row> {
    let root = row.position.root_wdl();
    let snapshot = &row.position.snapshot;
    let board = &snapshot.board;
    let black_to_move = !row.position.ply.is_multiple_of(2);
    let mut features = Vec::with_capacity(SQUARES * INPUTS);
    encode_position(
        &mut features,
        board,
        black_to_move,
        snapshot.castling,
        snapshot.ep_square,
        snapshot.halfmove_clock,
        snapshot.repetitions,
    )
    .ok_or("invalid piece code")?;
    Ok(Row {
        game: row.game_id,
        black: black_to_move,
        material: chess_network::hints::material_key(board, black_to_move),
        outcome: row.value,
        hint: [0.; chess_network::hints::FEATURES],
        features,
        reply: None,
        total_visits: moves.iter().map(|(_, visits)| visits).sum(),
        moves,
        value: std::array::from_fn(|i| (1. - root_fraction) * outcome[i] + root_fraction * root[i]),
        use_value,
    })
}

pub fn attach_hints(
    datasets: &mut [Dataset],
    config: chess_network::hints::HintConfig,
) -> chess_network::hints::MaterialTable {
    use chess_network::hints::{HintConfig, MaterialTable, TableBuilder};
    if matches!(config, HintConfig::Disabled {}) {
        return MaterialTable::default();
    }
    let mut builder = TableBuilder::default();
    for (phase, data) in datasets.iter().enumerate() {
        for row in &data.train {
            builder.observe(phase, row.game, row.material, row.black, row.outcome);
        }
    }
    let table = builder.finish();
    for data in datasets {
        for row in data.train.iter_mut().chain(&mut data.test) {
            row.hint = table.lookup(&row.material, config);
        }
    }
    table
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
        hint_dropout: f64,
        rng: &mut impl Rng,
    ) -> candle_core::Result<Self> {
        let size = rows.len();
        let mut positions = Vec::with_capacity(size * SQUARES * INPUTS);
        for row in rows.clone() {
            positions.extend_from_slice(&row.features);
        }
        let moves = rows.clone().map(|row| row.moves.iter().map(|&(mv, _)| mv));
        let inputs = PolicyInputs::new(positions, moves, device)?.with_hints(
            rows.clone()
                .map(|row| {
                    if hint_dropout > 0. && rng.random_bool(hint_dropout) {
                        [0.; chess_network::hints::FEATURES]
                    } else {
                        row.hint
                    }
                })
                .collect(),
        )?;
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
                policy[index] = visits / row.total_visits;
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
        let mut rescorer = crate::config::RescoringConfig::Disabled.open()?;
        let outcome = [0., 1., 0.];
        for (fraction, target) in [
            (0.0, outcome),
            (0.25, [0.03125, 0.875, 0.09375]),
            (0.5, [0.0625, 0.75, 0.1875]),
            (1.0, [0.125, 0.5, 0.375]),
        ] {
            config.root_wdl_fraction = fraction;
            let data = load(dir.path(), &config, 12, true, &mut rescorer)?;
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

    #[test]
    fn hints_use_selected_training_games_only() {
        use chess_network::hints::HintConfig;
        let row = |game, material, outcome| Row {
            game,
            material,
            outcome,
            black: false,
            hint: [0.; chess_network::hints::FEATURES],
            features: Vec::new(),
            moves: Vec::new(),
            reply: None,
            total_visits: 0.,
            value: crate::rescoring::game_value(outcome),
            use_value: false,
        };
        let mut datasets = [Dataset {
            train: (0..20).map(|game| row(game, [1; 10], 1)).collect(),
            test: (0..100)
                .map(|game| row(game, [2; 10], -1))
                .chain(std::iter::once(row(100, [1; 10], -1)))
                .collect(),
            target_seconds: 0.,
            train_games: 20,
            value_labels_per_game: Vec::new(),
            test_ids: Vec::new(),
            estimated_generation_seconds: 0.0,
        }];
        let config = HintConfig::Material {
            min_games: 20.try_into().unwrap(),
            injection: chess_network::hints::Injection::Early,
            dropout: 0.,
        };
        let table = attach_hints(&mut datasets, config);
        assert_eq!(
            table.lookup(&[2; 10], config),
            [0.; chess_network::hints::FEATURES]
        );
        let expected = table.lookup(&[1; 10], config);
        assert_eq!(expected[..3], [1. / 23., 1. / 23., 21. / 23.]);
        assert_eq!(datasets[0].test.last().unwrap().hint, expected);
        assert_eq!(datasets[0].train[0].hint, expected);
    }

    #[test]
    #[ignore = "requires downloaded ChessTB tables on the HDD"]
    fn rescoring_only_changes_selected_training_value_targets() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let positions: Vec<_> = (0..2)
            .map(|ply| {
                let mut board = [0; 32];
                for (square, code) in [(0, 6), (9, 5), (47, 12)] {
                    board[square / 2] |= code << (4 * (square % 2));
                }
                chess_data::Position {
                    ply,
                    snapshot: chess_data::Board {
                        board,
                        castling: 0,
                        ep_square: 64,
                        halfmove_clock: 0,
                        repetitions: 1,
                    },
                    policy: vec![(123, 800)],
                    root_q: 0.,
                    root_d: 0.5,
                    played_move: 123,
                }
            })
            .collect();
        chess_data::write_shard(
            directory.path(),
            0,
            (0..4).map(|id| (id, positions.as_slice(), 0)),
            42,
        )?;
        let mut disabled = crate::config::RescoringConfig::Disabled.open()?;
        let mut enabled = crate::config::RescoringConfig::Chesstb {
            directory: "/data/chess/tablebases/chesstb-5".into(),
            policy_mix: 0.0,
            value_mix: 1.0,
        }
        .open()?;
        use chess_config::TomlConfig;
        let mut config = crate::config::Config::from_repo_file("training.toml")?;
        config.value_sample_rate = 0.5;
        let original = load(directory.path(), &config, 6, true, &mut disabled)?;
        let rescored = load(directory.path(), &config, 6, true, &mut enabled)?;
        assert_eq!(original.test_ids, rescored.test_ids);
        assert_eq!(
            original.estimated_generation_seconds,
            rescored.estimated_generation_seconds
        );
        assert_eq!(
            original.value_labels_per_game,
            rescored.value_labels_per_game
        );
        assert_eq!(enabled.stats.probed, 3);
        assert_eq!(enabled.stats.changed, 3);
        for (before, after) in original.train.iter().zip(&rescored.train) {
            assert_eq!(before.features, after.features);
            assert_eq!(before.moves, after.moves);
            assert_eq!(before.total_visits, after.total_visits);
            assert_eq!(before.use_value, after.use_value);
            assert_eq!(before.value != after.value, after.use_value);
        }
        for (before, after) in original.test.iter().zip(&rescored.test) {
            assert_eq!(before.features, after.features);
            assert_eq!(before.value, after.value);
        }
        Ok(())
    }
}

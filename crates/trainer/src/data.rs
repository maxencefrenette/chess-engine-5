use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use candle_core::{Device, Tensor};
pub use chess_data::Result;
use chess_data::{discover_shards, read_shard};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

use chess_network::{
    INPUTS, SQUARES, encode_position,
    policy_inputs::PolicyInputs,
    value::{PIECE_FEATURES, feature_tensor, piece_features},
};

pub struct Row {
    features: Vec<f32>,
    moves: Vec<(u16, u16)>,
    total_visits: f32,
    pub value: u32,
    use_value: bool,
    piece_features: [f32; PIECE_FEATURES],
}

pub struct Dataset {
    pub train: Vec<Row>,
    pub test: Vec<Row>,
    pub train_games: usize,
    pub value_labels_per_game: Vec<usize>,
    pub test_ids: Vec<u64>,
    pub estimated_generation_seconds: Option<f64>,
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
    seed: u64,
    train_positions: usize,
    value_sample_rate: f64,
    retain_holdout: bool,
) -> Result<Dataset> {
    let mut rows = Vec::new();
    for shard in discover_shards(dataset)? {
        read_shard(&shard, |row| {
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
        .map(|row| row.generation_ns_per_position.map(u128::from))
        .sum::<Option<u128>>()
        .map(|ns| ns as f64 / 1e9);
    Ok(Dataset {
        train: train
            .into_iter()
            .zip(mask)
            .map(|(row, use_value)| encode_row(row, use_value))
            .collect::<Result<_>>()?,
        test: test
            .into_iter()
            .filter(|_| retain_holdout)
            .map(|row| encode_row(row, true))
            .collect::<Result<_>>()?,
        train_games: by_game.len(),
        value_labels_per_game,
        test_ids,
        estimated_generation_seconds,
    })
}

fn encode_row(row: chess_data::Row, use_value: bool) -> Result<Row> {
    let snapshot = row.position.snapshot;
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
        features,
        total_visits: row
            .position
            .policy
            .iter()
            .map(|&(_, visits)| f32::from(visits))
            .sum(),
        moves: row.position.policy,
        value: (row.value + 1) as u32,
        use_value,
        piece_features: piece_features(board, black_to_move),
    })
}

pub struct Batch {
    pub inputs: PolicyInputs,
    pub mask: Tensor,
    pub policy: Tensor,
    pub value: Option<Tensor>,
    pub value_indices: Option<Tensor>,
    pub value_count: usize,
    pub piece_features: Option<Tensor>,
    pub uniform_loss: f64,
}

impl Batch {
    pub fn new(
        rows: &[Row],
        indices: &[usize],
        device: &Device,
        uses_piece_counts: bool,
    ) -> candle_core::Result<Self> {
        let size = indices.len();
        let mut positions = Vec::with_capacity(size * SQUARES * INPUTS);
        for &i in indices {
            positions.extend_from_slice(&rows[i].features);
        }
        let moves = indices
            .iter()
            .map(|&i| rows[i].moves.iter().map(|&(mv, _)| mv));
        let inputs = PolicyInputs::new(positions, moves, device)?;
        let width = inputs.move_width();
        let mut mask = vec![-1e9_f32; size * width];
        let mut policy = vec![0f32; size * width];
        let mut value = Vec::with_capacity(size);
        let mut value_indices = Vec::new();
        let mut uniform_loss = 0.;
        for (i, &row) in indices.iter().enumerate() {
            let row = &rows[row];
            if row.use_value {
                value.push(row.value);
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
        let value_count = value.len();
        Ok(Self {
            inputs,
            mask: Tensor::from_vec(mask, shape, device)?,
            policy: Tensor::from_vec(policy, shape, device)?,
            value: (value_count > 0)
                .then(|| Tensor::from_vec(value, value_count, device))
                .transpose()?,
            value_indices: if (1..size).contains(&value_count) {
                Some(Tensor::from_vec(value_indices, value_count, device)?)
            } else {
                None
            },
            value_count,
            piece_features: if uses_piece_counts {
                Some(feature_tensor(
                    &indices
                        .iter()
                        .map(|&i| rows[i].piece_features)
                        .collect::<Vec<_>>(),
                    device,
                )?)
            } else {
                None
            },
            uniform_loss,
        })
    }
}

use std::{collections::BTreeSet, path::Path};

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
    piece_features: [f32; PIECE_FEATURES],
}

pub struct Dataset {
    pub train: Vec<Row>,
    pub test: Vec<Row>,
    pub train_games: usize,
    pub test_ids: Vec<u64>,
    pub estimated_generation_seconds: Option<f64>,
}

pub fn load(dataset: &Path, seed: u64, train_positions: usize) -> Result<Dataset> {
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
    let train_games = train
        .iter()
        .map(|row| row.game_id)
        .collect::<BTreeSet<_>>()
        .len();
    let estimated_generation_seconds = train
        .iter()
        .map(|row| row.generation_ns_per_position.map(u128::from))
        .sum::<Option<u128>>()
        .map(|ns| ns as f64 / 1e9);
    Ok(Dataset {
        train: train.into_iter().map(encode_row).collect::<Result<_>>()?,
        test: test.into_iter().map(encode_row).collect::<Result<_>>()?,
        train_games,
        test_ids,
        estimated_generation_seconds,
    })
}

fn encode_row(row: chess_data::Row) -> Result<Row> {
    let snapshot = row.position.snapshot;
    let board = &snapshot.board;
    let black_to_move = !row.ply.is_multiple_of(2);
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
        piece_features: piece_features(board, black_to_move),
    })
}

pub struct Batch {
    pub inputs: PolicyInputs,
    pub mask: Tensor,
    pub policy: Tensor,
    pub value: Tensor,
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
        let mut uniform_loss = 0.;
        for (i, &row) in indices.iter().enumerate() {
            let row = &rows[row];
            value.push(row.value);
            uniform_loss += (row.moves.len() as f64).ln();
            for (j, &(_, visits)) in row.moves.iter().enumerate() {
                let index = i * width + j;
                mask[index] = 0.;
                policy[index] = f32::from(visits) / row.total_visits;
            }
        }
        let shape = (size, width);
        Ok(Self {
            inputs,
            mask: Tensor::from_vec(mask, shape, device)?,
            policy: Tensor::from_vec(policy, shape, device)?,
            value: Tensor::from_vec(value, size, device)?,
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

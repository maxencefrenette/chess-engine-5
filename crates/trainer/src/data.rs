use std::{collections::BTreeSet, fs, path::Path};

use candle_core::{Device, Tensor};
pub use chess_data::Result;
use chess_data::read_shard;
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

use chess_network::{
    INPUTS, SQUARES, encode_position, move_indices,
    piece_value::{PIECE_FEATURES, feature_tensor, piece_features},
};

pub struct Row {
    game_id: u64,
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
}

pub fn load(dataset: &Path, seed: u64) -> Result<Dataset> {
    let shards = if dataset.is_dir() {
        let mut shards = Vec::new();
        for entry in fs::read_dir(dataset)? {
            let path = entry?.path();
            if path.extension().is_some_and(|ext| ext == "parquet") && path.is_file() {
                shards.push(path);
            }
        }
        shards.sort_unstable();
        shards
    } else {
        vec![dataset.into()]
    };
    let mut rows = Vec::new();
    for shard in shards {
        read_shard(&shard, |row| {
            let snapshot = row.position.snapshot;
            let board = &snapshot.board;
            let black_to_move = row.ply % 2 != 0;
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
            rows.push(Row {
                game_id: row.game_id,
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
            });
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
    let (test, train) = rows
        .into_iter()
        .partition(|row| test_ids.binary_search(&row.game_id).is_ok());
    Ok(Dataset {
        train,
        test,
        train_games: games.len() - test_games,
        test_ids,
    })
}

pub struct Batch {
    pub position: Tensor,
    pub move_index: Tensor,
    pub promotion_index: Tensor,
    pub promotes: Tensor,
    pub mask: Tensor,
    pub policy: Tensor,
    pub value: Tensor,
    pub piece_features: Tensor,
    pub uniform_loss: f64,
}

impl Batch {
    pub fn new(rows: &[Row], indices: &[usize], device: &Device) -> candle_core::Result<Self> {
        let size = indices.len();
        let width = indices.iter().map(|&i| rows[i].moves.len()).max().unwrap();
        let mut positions = Vec::with_capacity(size * SQUARES * INPUTS);
        let mut move_index = vec![0u32; size * width];
        let mut promotion_index = move_index.clone();
        let mut promotes = vec![0f32; size * width];
        let mut mask = vec![-1e9_f32; size * width];
        let mut policy = vec![0f32; size * width];
        let mut value = Vec::with_capacity(size);
        let mut uniform_loss = 0.;
        for (i, &row) in indices.iter().enumerate() {
            let row = &rows[row];
            positions.extend_from_slice(&row.features);
            value.push(row.value);
            uniform_loss += (row.moves.len() as f64).ln();
            for (j, &(mv, visits)) in row.moves.iter().enumerate() {
                let index = i * width + j;
                (move_index[index], promotion_index[index], promotes[index]) = move_indices(mv);
                mask[index] = 0.;
                policy[index] = f32::from(visits) / row.total_visits;
            }
        }
        let shape = (size, width);
        Ok(Self {
            position: Tensor::from_vec(positions, (size, SQUARES, INPUTS), device)?,
            move_index: Tensor::from_vec(move_index, shape, device)?,
            promotion_index: Tensor::from_vec(promotion_index, shape, device)?,
            promotes: Tensor::from_vec(promotes, shape, device)?,
            mask: Tensor::from_vec(mask, shape, device)?,
            policy: Tensor::from_vec(policy, shape, device)?,
            value: Tensor::from_vec(value, size, device)?,
            piece_features: feature_tensor(
                &indices
                    .iter()
                    .map(|&i| rows[i].piece_features)
                    .collect::<Vec<_>>(),
                device,
            )?,
            uniform_loss,
        })
    }
}

use std::{collections::BTreeSet, error::Error, fs::File, path::Path};

use arrow_array::{
    Array, BinaryArray, FixedSizeBinaryArray, Int8Array, UInt8Array, UInt16Array, UInt64Array,
};
use candle_core::{Device, Tensor};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

use chess_network::{INPUTS, SQUARES, encode_position, move_indices};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub struct Row {
    game_id: u64,
    features: Vec<f32>,
    moves: Vec<(u16, u16)>,
    pub value: u32,
    material: f32,
}

pub struct Dataset {
    pub train: Vec<Row>,
    pub test: Vec<Row>,
    pub train_games: usize,
    pub test_ids: Vec<u64>,
}

pub fn load(shard: &Path, seed: u64) -> Result<Dataset> {
    let mut rows = Vec::new();
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(shard)?)?.build()?;
    for batch in reader {
        let batch = batch?;
        let column = |name: &str| {
            batch
                .column_by_name(name)
                .filter(|column| column.null_count() == 0)
                .ok_or_else(|| format!("missing or nullable data in {name}"))
        };
        macro_rules! typed {
            ($name:literal, $ty:ty) => {
                column($name)?
                    .as_any()
                    .downcast_ref::<$ty>()
                    .ok_or(concat!("invalid Parquet type for ", $name))?
            };
        }
        let ids = typed!("game_id", UInt64Array);
        let ply = typed!("ply", UInt16Array);
        let board = typed!("board", FixedSizeBinaryArray);
        let castling = typed!("castling", UInt8Array);
        let ep = typed!("ep_square", UInt8Array);
        let clock = typed!("halfmove_clock", UInt8Array);
        let repetitions = typed!("repetitions", UInt8Array);
        let policy = typed!("policy_moves", BinaryArray);
        let visits = typed!("policy_visits", BinaryArray);
        let values = typed!("value", Int8Array);
        if board.value_length() != 32 {
            return Err("board must be 32 bytes".into());
        }
        for i in 0..batch.num_rows() {
            let moves = decode_policy(policy.value(i), visits.value(i))?;
            let board: &[u8; 32] = board.value(i).try_into()?;
            let black_to_move = ply.value(i) % 2 != 0;
            let mut features = Vec::with_capacity(SQUARES * INPUTS);
            encode_position(
                &mut features,
                board,
                black_to_move,
                castling.value(i),
                ep.value(i),
                clock.value(i),
                repetitions.value(i),
            )
            .ok_or("invalid piece code")?;
            const MATERIAL: [i32; 13] = [0, 1, 3, 3, 5, 9, 0, -1, -3, -3, -5, -9, 0];
            let white_material: i32 = board
                .iter()
                .map(|byte| MATERIAL[(byte & 15) as usize] + MATERIAL[(byte >> 4) as usize])
                .sum();
            let value = match values.value(i) {
                -1 => 0,
                0 => 1,
                1 => 2,
                _ => return Err("invalid game result".into()),
            };
            let material = if ply.value(i) % 2 == 0 {
                white_material
            } else {
                -white_material
            } as f32;
            rows.push(Row {
                game_id: ids.value(i),
                features,
                moves,
                value,
                material: material / (material.abs() + 8.0),
            });
        }
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

fn decode_policy(moves: &[u8], visits: &[u8]) -> Result<Vec<(u16, u16)>> {
    if moves.is_empty() || moves.len() != visits.len() || !moves.len().is_multiple_of(2) {
        return Err("invalid packed policy lengths".into());
    }
    let pairs: Vec<_> = moves
        .chunks_exact(2)
        .zip(visits.chunks_exact(2))
        .map(|(mv, count)| {
            (
                u16::from_le_bytes([mv[0], mv[1]]),
                u16::from_le_bytes([count[0], count[1]]),
            )
        })
        .collect();
    if pairs.iter().map(|&(_, n)| u32::from(n)).sum::<u32>() != 800
        || pairs.iter().any(|&(mv, _)| mv >> 12 > 4)
    {
        return Err("invalid move encoding or visit total".into());
    }
    Ok(pairs)
}

pub struct Batch {
    pub position: Tensor,
    pub move_index: Tensor,
    pub promotion_index: Tensor,
    pub promotes: Tensor,
    pub material: Option<Tensor>,
    pub mask: Tensor,
    pub policy: Tensor,
    pub value: Tensor,
    pub uniform_loss: f64,
}

impl Batch {
    pub fn new(
        rows: &[Row],
        indices: &[usize],
        device: &Device,
        material_hint: bool,
    ) -> candle_core::Result<Self> {
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
                policy[index] = f32::from(visits) / 800.;
            }
        }
        let shape = (size, width);
        Ok(Self {
            position: Tensor::from_vec(positions, (size, SQUARES, INPUTS), device)?,
            move_index: Tensor::from_vec(move_index, shape, device)?,
            promotion_index: Tensor::from_vec(promotion_index, shape, device)?,
            promotes: Tensor::from_vec(promotes, shape, device)?,
            material: material_hint
                .then(|| {
                    let values: Vec<_> = indices.iter().map(|&i| rows[i].material).collect();
                    Tensor::from_vec(values, (size, 1), device)
                })
                .transpose()?,
            mask: Tensor::from_vec(mask, shape, device)?,
            policy: Tensor::from_vec(policy, shape, device)?,
            value: Tensor::from_vec(value, size, device)?,
            uniform_loss,
        })
    }
}

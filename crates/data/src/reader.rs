use std::{
    fs::{self, File},
    io,
    path::{Path, PathBuf},
};

use arrow_array::{
    BinaryArray, FixedSizeBinaryArray, Int8Array, UInt8Array, UInt16Array, UInt64Array,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::{Board, GENERATION_TIME_COLUMN, Position, Result, Row, schema};

/// Accept a single shard or list the Parquet files in a directory in path order.
pub fn discover_shards(path: &Path) -> io::Result<Vec<PathBuf>> {
    if !path.is_dir() {
        return Ok(vec![path.into()]);
    }
    let mut shards = Vec::new();
    for entry in fs::read_dir(path)? {
        let path = entry?.path();
        if path.extension().is_some_and(|ext| ext == "parquet") && path.is_file() {
            shards.push(path);
        }
    }
    shards.sort_unstable();
    Ok(shards)
}

/// Decode a shard one Arrow batch at a time, visiting rows in file order.
pub fn read_shard(path: &Path, mut visit: impl FnMut(Row) -> Result<()>) -> Result<()> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    let has_timing = builder.schema().index_of(GENERATION_TIME_COLUMN).is_ok();
    let expected = schema(has_timing);
    let mut indices = Vec::with_capacity(expected.fields().len());
    for field in expected.fields() {
        let index = builder.schema().index_of(field.name())?;
        if builder.schema().field(index).data_type() != field.data_type() {
            return Err(format!("invalid Parquet type for {}", field.name()).into());
        }
        indices.push(index);
    }

    for batch in builder.build()? {
        let batch = batch?;
        let columns = indices.iter().map(|&index| batch.column(index));
        for (field, column) in expected.fields().iter().zip(columns.clone()) {
            if column.null_count() != 0 {
                return Err(format!("null data in {}", field.name()).into());
            }
        }
        let mut columns = columns.enumerate();
        macro_rules! next {
            ($ty:ty) => {{
                let (i, column) = columns.next().expect("schema column exists");
                column.as_any().downcast_ref::<$ty>().ok_or_else(|| {
                    format!("invalid Arrow array for {}", expected.field(i).name())
                })?
            }};
        }
        let ids = next!(UInt64Array);
        let ply = next!(UInt16Array);
        let board = next!(FixedSizeBinaryArray);
        let castling = next!(UInt8Array);
        let ep = next!(UInt8Array);
        let clock = next!(UInt8Array);
        let repetitions = next!(UInt8Array);
        let moves = next!(BinaryArray);
        let visits = next!(BinaryArray);
        let played_move = next!(UInt16Array);
        let values = next!(Int8Array);
        let timing = if has_timing {
            Some(next!(UInt64Array))
        } else {
            None
        };

        for i in 0..batch.num_rows() {
            let value = values.value(i);
            if !(-1..=1).contains(&value) {
                return Err("invalid game result".into());
            }
            visit(Row {
                game_id: ids.value(i),
                value,
                generation_ns_per_position: timing.map(|column| column.value(i)),
                position: Position {
                    ply: ply.value(i),
                    snapshot: Board {
                        board: board.value(i).try_into()?,
                        castling: castling.value(i),
                        ep_square: ep.value(i),
                        halfmove_clock: clock.value(i),
                        repetitions: repetitions.value(i),
                    },
                    policy: decode_policy(moves.value(i), visits.value(i))?,
                    played_move: played_move.value(i),
                },
            })?;
        }
    }
    Ok(())
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
    let total = pairs.iter().map(|&(_, n)| u64::from(n)).sum::<u64>();
    if total == 0 || total > u64::from(u16::MAX) || pairs.iter().any(|&(mv, _)| mv >> 12 > 4) {
        return Err("invalid move encoding or visit total".into());
    }
    Ok(pairs)
}

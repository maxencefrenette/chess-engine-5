use std::{
    fs::{self, File},
    path::Path,
    sync::Arc,
};

use arrow_array::{
    Int8Array, RecordBatch, UInt8Array, UInt16Array, UInt64Array,
    builder::{BinaryBuilder, FixedSizeBinaryBuilder},
};
use arrow_schema::SchemaRef;
use parquet::{arrow::ArrowWriter, basic::Compression, file::properties::WriterProperties};

use crate::{Position, Result, schema};

const ROW_GROUP_ROWS: usize = 65_536;

/// Write whole games to one shard and publish it once the Parquet footer is complete.
pub fn write_shard<'a>(
    directory: &Path,
    index: usize,
    games: impl IntoIterator<Item = (u64, &'a [Position], i8)>,
    generation_ns_per_position: u64,
) -> Result<()> {
    let schema = schema(true);
    let path = directory.join(format!("part-{index:05}.parquet"));
    let partial = path.with_extension("parquet.partial");
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .set_max_row_group_row_count(Some(ROW_GROUP_ROWS))
        .build();
    let mut writer =
        ArrowWriter::try_new(File::create_new(&partial)?, schema.clone(), Some(props))?;
    for (id, positions, value) in games {
        if positions.is_empty() || !(-1..=1).contains(&value) {
            return Err("expected a non-empty game with a valid result".into());
        }
        writer.write(&encode_game(
            schema.clone(),
            id,
            positions,
            value,
            generation_ns_per_position,
        )?)?;
    }
    writer.close()?;
    fs::rename(partial, path)?;
    Ok(())
}

fn encode_game(
    schema: SchemaRef,
    game_id: u64,
    rows: &[Position],
    white_value: i8,
    generation_ns_per_position: u64,
) -> Result<RecordBatch> {
    let mut board = FixedSizeBinaryBuilder::with_capacity(rows.len(), 32);
    let mut policy_moves = BinaryBuilder::new();
    let mut policy_visits = BinaryBuilder::new();
    let mut moves = Vec::new();
    let mut visits = Vec::new();
    for row in rows {
        board.append_value(row.snapshot.board)?;
        moves.clear();
        visits.clear();
        for &(mv, count) in &row.policy {
            moves.extend(mv.to_le_bytes());
            visits.extend(count.to_le_bytes());
        }
        policy_moves.append_value(&moves);
        policy_visits.append_value(&visits);
    }
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(vec![game_id; rows.len()])),
            Arc::new(UInt16Array::from_iter_values(
                rows.iter().map(|row| row.ply),
            )),
            Arc::new(board.finish()),
            Arc::new(UInt8Array::from_iter_values(
                rows.iter().map(|row| row.snapshot.castling),
            )),
            Arc::new(UInt8Array::from_iter_values(
                rows.iter().map(|row| row.snapshot.ep_square),
            )),
            Arc::new(UInt8Array::from_iter_values(
                rows.iter().map(|row| row.snapshot.halfmove_clock),
            )),
            Arc::new(UInt8Array::from_iter_values(
                rows.iter().map(|row| row.snapshot.repetitions),
            )),
            Arc::new(policy_moves.finish()),
            Arc::new(policy_visits.finish()),
            Arc::new(UInt16Array::from_iter_values(
                rows.iter().map(|row| row.played_move),
            )),
            Arc::new(Int8Array::from_iter_values(rows.iter().map(|row| {
                if row.ply % 2 == 0 {
                    white_value
                } else {
                    -white_value
                }
            }))),
            Arc::new(UInt64Array::from(vec![
                generation_ns_per_position;
                rows.len()
            ])),
        ],
    )?)
}

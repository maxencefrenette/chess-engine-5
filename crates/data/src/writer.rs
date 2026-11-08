use std::{
    fs::{self, File},
    path::{Path, PathBuf},
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
const SHARD_ROWS: usize = 4 * ROW_GROUP_ROWS;

/// A Parquet run writer that keeps games whole and publishes each closed shard.
pub struct RunWriter {
    directory: PathBuf,
    schema: SchemaRef,
    shard: Option<Shard>,
    shards: usize,
}

impl RunWriter {
    pub fn new(directory: &Path) -> Self {
        Self {
            directory: directory.into(),
            schema: schema(),
            shard: None,
            shards: 0,
        }
    }

    pub fn write_game(
        &mut self,
        game_id: u64,
        positions: &[Position],
        white_value: i8,
    ) -> Result<()> {
        if positions.is_empty() || !(-1..=1).contains(&white_value) {
            return Err("expected a non-empty game with a valid result".into());
        }
        let batch = encode_game(self.schema.clone(), game_id, positions, white_value)?;
        if self
            .shard
            .as_ref()
            .is_some_and(|shard| shard.rows + batch.num_rows() > SHARD_ROWS)
        {
            self.shard.take().unwrap().finish()?;
        }
        if self.shard.is_none() {
            self.shard = Some(Shard::create(
                &self.directory,
                self.shards,
                self.schema.clone(),
            )?);
            self.shards += 1;
        }
        let shard = self.shard.as_mut().unwrap();
        shard.writer.write(&batch)?;
        shard.rows += batch.num_rows();
        Ok(())
    }

    pub fn finish(self) -> Result<usize> {
        if let Some(shard) = self.shard {
            shard.finish()?;
        }
        Ok(self.shards)
    }
}

fn encode_game(
    schema: SchemaRef,
    game_id: u64,
    rows: &[Position],
    white_value: i8,
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
            Arc::new(UInt16Array::from(
                (0..u16::try_from(rows.len())?).collect::<Vec<_>>(),
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
            Arc::new(Int8Array::from_iter_values((0..rows.len()).map(|ply| {
                if ply % 2 == 0 {
                    white_value
                } else {
                    -white_value
                }
            }))),
        ],
    )?)
}

struct Shard {
    writer: ArrowWriter<File>,
    path: PathBuf,
    partial: PathBuf,
    rows: usize,
}

impl Shard {
    fn create(run: &Path, index: usize, schema: SchemaRef) -> Result<Self> {
        let path = run.join(format!("part-{index:05}.parquet"));
        let partial = path.with_extension("parquet.partial");
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .set_max_row_group_row_count(Some(ROW_GROUP_ROWS))
            .build();
        let writer = ArrowWriter::try_new(File::create_new(&partial)?, schema, Some(props))?;
        Ok(Self {
            writer,
            path,
            partial,
            rows: 0,
        })
    }

    fn finish(self) -> Result<()> {
        self.writer.close()?;
        fs::rename(self.partial, self.path)?;
        Ok(())
    }
}

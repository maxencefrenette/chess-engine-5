//! Chess Engine 5's Parquet format, independent of search and training.

mod reader;
mod writer;

use std::{error::Error, sync::Arc};

use arrow_schema::{DataType, Field, Schema, SchemaRef};

pub use reader::{discover_shards, read_shard};
pub use writer::write_shard;

pub const FORMAT_VERSION: u32 = 2;
const GENERATION_TIME_COLUMN: &str = "generation_ns_per_position";

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug, PartialEq, Eq)]
pub struct Board {
    pub board: [u8; 32],
    pub castling: u8,
    pub ep_square: u8,
    pub halfmove_clock: u8,
    pub repetitions: u8,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Position {
    pub ply: u16,
    pub snapshot: Board,
    /// Packed move IDs and their visit counts, in matching order.
    pub policy: Vec<(u16, u16)>,
    pub played_move: u16,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Row {
    pub game_id: u64,
    pub position: Position,
    /// Final result from the side-to-move perspective.
    pub value: i8,
    /// Shard generation wall time divided by its position count, absent in v1.
    pub generation_ns_per_position: Option<u64>,
}

fn schema(has_timing: bool) -> SchemaRef {
    let mut fields = vec![
        Field::new("game_id", DataType::UInt64, false),
        Field::new("ply", DataType::UInt16, false),
        Field::new("board", DataType::FixedSizeBinary(32), false),
        Field::new("castling", DataType::UInt8, false),
        Field::new("ep_square", DataType::UInt8, false),
        Field::new("halfmove_clock", DataType::UInt8, false),
        Field::new("repetitions", DataType::UInt8, false),
        Field::new("policy_moves", DataType::Binary, false),
        Field::new("policy_visits", DataType::Binary, false),
        Field::new("played_move", DataType::UInt16, false),
        Field::new("value", DataType::Int8, false),
    ];
    if has_timing {
        fields.push(Field::new(GENERATION_TIME_COLUMN, DataType::UInt64, false));
    }
    Arc::new(Schema::new(fields))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_round_trip_preserves_positions_and_side_to_move_values() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let positions = [0, 3, 6].map(|ply| Position {
            ply,
            snapshot: Board {
                board: [ply as u8; 32],
                castling: 15 - ply as u8,
                ep_square: 64 - ply as u8,
                halfmove_clock: ply as u8,
                repetitions: 1,
            },
            policy: vec![(0x10ff, 15), (0x0402, 1), (0x0003, 0)],
            played_move: 0x10ff,
        });

        write_shard(dir.path(), 0, [(123, positions.as_slice(), 1)], 1234)?;

        let mut rows = Vec::new();
        read_shard(&dir.path().join("part-00000.parquet"), |row| {
            rows.push(row);
            Ok(())
        })?;
        assert_eq!(rows.len(), positions.len());
        for (row, position) in rows.into_iter().zip(positions) {
            assert_eq!(row.game_id, 123);
            assert_eq!(row.value, if position.ply % 2 == 0 { 1 } else { -1 });
            assert_eq!(row.position, position);
            assert_eq!(row.generation_ns_per_position, Some(1234));
        }
        Ok(())
    }
}

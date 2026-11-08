//! Chess Engine 5's Parquet v1 format, independent of search and training.

mod reader;
mod writer;

use std::{error::Error, sync::Arc};

use arrow_schema::{DataType, Field, Schema, SchemaRef};

pub use reader::read_shard;
pub use writer::RunWriter;

pub const FORMAT_VERSION: u32 = 1;
pub const VISITS_PER_MOVE: u16 = 800;

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
    pub snapshot: Board,
    /// Packed move IDs and their visit counts, in matching order.
    pub policy: Vec<(u16, u16)>,
    pub played_move: u16,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Row {
    pub game_id: u64,
    pub ply: u16,
    pub position: Position,
    /// Final result from the side-to-move perspective.
    pub value: i8,
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
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
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_round_trip_preserves_positions_and_side_to_move_values() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let positions = [0, 1].map(|ply| Position {
            snapshot: Board {
                board: [ply; 32],
                castling: 15 - ply,
                ep_square: 64 - ply,
                halfmove_clock: ply,
                repetitions: 1,
            },
            policy: vec![(0x10ff, VISITS_PER_MOVE - 1), (0x0402, 1), (0x0003, 0)],
            played_move: 0x10ff,
        });

        let mut writer = RunWriter::new(dir.path());
        writer.write_game(123, &positions, 1)?;
        assert_eq!(writer.finish()?, 1);

        let mut rows = Vec::new();
        read_shard(&dir.path().join("part-00000.parquet"), |row| {
            rows.push(row);
            Ok(())
        })?;
        assert_eq!(rows.len(), positions.len());
        for (ply, (row, position)) in rows.into_iter().zip(positions).enumerate() {
            assert_eq!(row.game_id, 123);
            assert_eq!(row.ply, ply as u16);
            assert_eq!(row.value, if ply == 0 { 1 } else { -1 });
            assert_eq!(row.position, position);
        }
        Ok(())
    }
}

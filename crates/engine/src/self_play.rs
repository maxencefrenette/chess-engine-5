use std::{
    env,
    error::Error,
    fs::{self, File},
    num::{NonZeroU32, NonZeroUsize},
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow_array::{
    Int8Array, RecordBatch, UInt8Array, UInt16Array, UInt64Array,
    builder::{BinaryBuilder, FixedSizeBinaryBuilder},
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet::{arrow::ArrowWriter, basic::Compression, file::properties::WriterProperties};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde_json::json;

use crate::{
    chess::{Game, Outcome, Snapshot},
    mcts::{MoveVisits, material_value, search},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const VISITS: NonZeroU32 = NonZeroU32::new(800).unwrap();
const ROW_GROUP_ROWS: usize = 65_536;
const SHARD_ROWS: usize = 4 * ROW_GROUP_ROWS;
const GAME_BATCH: usize = 16;

struct PositionRow {
    snapshot: Snapshot,
    policy: Vec<MoveVisits>,
    played_move: u16,
}

struct PlayedGame {
    positions: Vec<PositionRow>,
    white_value: i8,
}

fn play_game(seed: u64, game_id: u64) -> PlayedGame {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(game_id);
    let mut game = Game::default();
    let mut positions = Vec::new();
    loop {
        let policy = search(&game, VISITS, material_value);
        if policy.is_empty() {
            let white_value = match game.outcome().expect("empty search means game over") {
                Outcome::Win(crate::chess::Color::White) => 1,
                Outcome::Win(_) => -1,
                Outcome::Draw => 0,
            };
            return PlayedGame {
                positions,
                white_value,
            };
        }

        // Temperature 1 samples each move in direct proportion to its visits.
        let mut ticket = rng.random_range(0..VISITS.get());
        let played = policy
            .iter()
            .find(|entry| {
                if ticket < entry.visits {
                    true
                } else {
                    ticket -= entry.visits;
                    false
                }
            })
            .expect("root visits sum to the search budget")
            .mv;
        let legal_moves: Vec<_> = policy.iter().map(|entry| entry.mv).collect();
        positions.push(PositionRow {
            snapshot: game.snapshot(&legal_moves),
            policy,
            played_move: played.packed(),
        });
        game.play_legal(played);
    }
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

fn encode_game(schema: SchemaRef, game_id: u64, game: &PlayedGame) -> Result<RecordBatch> {
    let rows = &game.positions;
    let mut board = FixedSizeBinaryBuilder::with_capacity(rows.len(), 32);
    let mut policy_moves = BinaryBuilder::new();
    let mut policy_visits = BinaryBuilder::new();
    let mut moves = Vec::new();
    let mut visits = Vec::new();
    for row in rows {
        board.append_value(row.snapshot.board)?;
        moves.clear();
        visits.clear();
        for entry in &row.policy {
            moves.extend(entry.mv.packed().to_le_bytes());
            visits.extend(u16::try_from(entry.visits)?.to_le_bytes());
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
                    game.white_value
                } else {
                    -game.white_value
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

pub fn generate(root: &Path, games: NonZeroUsize, seed: u64) -> Result<PathBuf> {
    let version = root.join("v1");
    fs::create_dir_all(&version)?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let run = version.join(format!("{now}-{}", std::process::id()));
    fs::create_dir(&run)?;
    eprintln!("Writing self-play run to {}", run.display());

    let schema = schema();
    let mut shard_index = 0;
    let mut shard = Shard::create(&run, shard_index, schema.clone())?;
    let mut positions = 0;
    let mut white_wins = 0;
    let mut black_wins = 0;
    let mut draws = 0;
    for start in (0..games.get()).step_by(GAME_BATCH) {
        let end = start.saturating_add(GAME_BATCH).min(games.get());
        let batch: Vec<_> = (start..end)
            .into_par_iter()
            .map(|id| play_game(seed, id as u64))
            .collect();

        for (id, game) in (start..end).zip(batch) {
            let batch = encode_game(schema.clone(), id as u64, &game)?;
            if shard.rows > 0 && shard.rows + batch.num_rows() > SHARD_ROWS {
                shard.finish()?;
                shard_index += 1;
                shard = Shard::create(&run, shard_index, schema.clone())?;
            }
            shard.writer.write(&batch)?;
            shard.rows += batch.num_rows();
            positions += batch.num_rows();
            match game.white_value {
                1 => white_wins += 1,
                -1 => black_wins += 1,
                _ => draws += 1,
            }
        }
        eprintln!("{end}/{} games, {positions} positions", games.get());
    }
    shard.finish()?;

    let manifest = json!({
        "format_version": 1,
        "seed": seed,
        "engine_version": env!("CARGO_PKG_VERSION"),
        "engine_revision": env::var("CE5_ENGINE_REVISION").ok(),
        "games": games.get(),
        "positions": positions,
        "shards": shard_index + 1,
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "start_position": "standard",
        "visits_per_move": VISITS.get(),
        "evaluator": "material-v1",
        "priors": "uniform",
        "sampling": "chacha8-stream-per-game; probability proportional to root visits",
        "temperature": 1,
    });
    let partial = run.join("manifest.json.partial");
    fs::write(&partial, format!("{manifest:#}\n"))?;
    fs::rename(partial, run.join("manifest.json"))?;
    Ok(run)
}

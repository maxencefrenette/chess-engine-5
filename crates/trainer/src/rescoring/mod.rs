use crate::config::RescoringConfig;
use chess_data::{Board, Row};
use chess_tablebase::{RootMove, Tables, Wdl};
use serde::Serialize;
use std::{ffi::CString, io};

impl RescoringConfig {
    pub fn open(&self) -> io::Result<Rescorer> {
        Ok(Rescorer {
            tables: match self {
                Self::Disabled => None,
                Self::Chesstb { directory, .. } => {
                    let tables = Tables::open(directory)?;
                    if tables.largest() != 5 {
                        return Err(io::Error::other("expected tables through five pieces"));
                    }
                    Some(tables)
                }
            },
            policy_mix: match self {
                Self::Disabled => 0.,
                Self::Chesstb { policy_mix, .. } => policy_mix.get() as f32,
            },
            value_mix: match self {
                Self::Disabled => 0.,
                Self::Chesstb { value_mix, .. } => value_mix.get() as f32,
            },
            stats: Stats::default(),
        })
    }
}

#[derive(Default, Serialize)]
pub struct Stats {
    pub probed: usize,
    pub policy_boosted: usize,
    pub history_skipped: usize,
    pub changed: usize,
    /// Original target (rows) versus rescored target (columns), loss/draw/win.
    pub transitions: [[usize; 3]; 3],
}

pub struct Rescorer {
    tables: Option<Tables>,
    policy_mix: f32,
    value_mix: f32,
    pub stats: Stats,
}

impl Rescorer {
    /// Blend the search policy with uniform probability over selected DTZ moves.
    pub fn policy(&mut self, row: &Row) -> io::Result<Vec<(u16, f32)>> {
        let mut policy: Vec<_> = row
            .position
            .policy
            .iter()
            .map(|&(m, n)| (m, f32::from(n)))
            .collect();
        let board = &row.position.snapshot;
        if self.policy_mix == 0.
            || board.castling != 0
            || (0..64).filter(|&s| piece(board, s) != 0).count() > 5
        {
            return Ok(policy);
        }
        let Some(tables) = &mut self.tables else {
            return Ok(policy);
        };
        let fen = fen(board, !row.position.ply.is_multiple_of(2))?;
        let moves = tables.root_dtz(&fen, board.ep_square, board.halfmove_clock, true, true)?;
        let selected = boost_moves(&moves, board.halfmove_clock);
        if selected.iter().any(|m| !policy.iter().any(|(p, _)| p == m)) {
            return Err(io::Error::other(
                "tablebase move absent from legal policy targets",
            ));
        }
        if !selected.is_empty() {
            blend_policy(&mut policy, &selected, self.policy_mix);
            self.stats.policy_boosted += 1;
        }
        Ok(policy)
    }

    /// Mix game outcomes and tablebase outcomes in loss/draw/win order.
    pub fn value(&mut self, row: &mut Row, repeated: bool) -> io::Result<[f32; 3]> {
        let original = row.value;
        if self.value_mix > 0. {
            self.rescore_with_history(row, repeated)?;
        }
        Ok(blend_value(original, row.value, self.value_mix))
    }

    pub fn rescore_with_history(&mut self, row: &mut Row, repeated: bool) -> io::Result<()> {
        if repeated {
            if self.tables.is_some() {
                self.stats.history_skipped += 1;
            }
            return Ok(());
        }
        self.rescore(row)
    }

    /// Call only for selected training value labels; held-out labels stay intact.
    pub fn rescore(&mut self, row: &mut Row) -> io::Result<()> {
        let Some(tables) = &mut self.tables else {
            return Ok(());
        };
        let board = &row.position.snapshot;
        let pieces = (0..64).filter(|&square| piece(board, square) != 0).count();
        if pieces > 5 {
            return Ok(());
        }
        let fen = fen(board, !row.position.ply.is_multiple_of(2))?;
        let class = tables.wdl(&fen, board.ep_square, board.halfmove_clock)?;
        let value = if board.halfmove_clock >= 100 || board.repetitions >= 3 {
            0
        } else {
            wdl_target(class)
        };
        if !(-1..=1).contains(&row.value) {
            return Err(io::Error::other("invalid original WDL target"));
        }
        self.stats.probed += 1;
        self.stats.changed += usize::from(value != row.value);
        self.stats.transitions[(row.value + 1) as usize][(value + 1) as usize] += 1;
        row.value = value;
        Ok(())
    }
}

pub fn game_value(value: i8) -> [f32; 3] {
    let mut target = [0.; 3];
    target[(value + 1) as usize] = 1.;
    target
}

fn blend_value(original: i8, rescored: i8, mix: f32) -> [f32; 3] {
    let mut target = game_value(original).map(|p| p * (1. - mix));
    target[(rescored + 1) as usize] += mix;
    target
}

fn blend_policy(policy: &mut [(u16, f32)], selected: &[u16], mix: f32) {
    let total: f32 = policy.iter().map(|(_, p)| p).sum();
    for (m, p) in policy {
        *p *= 1. - mix;
        if selected.contains(m) {
            *p += total * mix / selected.len() as f32;
        }
    }
}

/// Full recorded history is needed before training-position subsampling.
pub struct History(std::collections::BTreeMap<u64, Vec<u16>>);
impl History {
    pub fn new(rows: &[Row]) -> Self {
        let mut games = std::collections::BTreeMap::<_, Vec<_>>::new();
        for row in rows {
            if row.position.snapshot.repetitions > 1 {
                games.entry(row.game_id).or_default().push(row.position.ply);
            }
        }
        for plies in games.values_mut() {
            plies.sort_unstable();
        }
        Self(games)
    }
    pub fn has_repeated(&self, row: &Row) -> bool {
        let ply = row.position.ply;
        self.0.get(&row.game_id).is_some_and(|plies| {
            let end = plies.partition_point(|&p| p <= ply);
            end > 0
                && plies[end - 1]
                    >= ply.saturating_sub(u16::from(row.position.snapshot.halfmove_clock))
        })
    }
}

fn wdl_target(class: Wdl) -> i8 {
    match class {
        Wdl::Loss => -1,
        Wdl::BlessedLoss | Wdl::Draw | Wdl::CursedWin => 0,
        Wdl::Win => 1,
    }
}

// Match Lc0's DTZ-only rescorer: has_repeated=true, win_only=true.
fn boost_moves(moves: &[RootMove], clock: u8) -> Vec<u16> {
    let rank = |m: &RootMove| 1000 - m.dtz - i32::from(clock);
    let best = moves
        .iter()
        .filter(|m| m.dtz > 0)
        .map(rank)
        .max()
        .unwrap_or(1)
        .max(1);
    moves
        .iter()
        .filter(|m| m.dtz > 0 && rank(m) == best)
        .map(|m| u16::from(m.from) | (u16::from(m.to) << 6) | (u16::from(m.promotion) << 12))
        .collect()
}

fn piece(board: &Board, square: usize) -> u8 {
    (board.board[square / 2] >> (4 * (square % 2))) & 15
}

fn fen(board: &Board, black: bool) -> io::Result<CString> {
    let mut text = String::with_capacity(90);
    for rank in (0..8).rev() {
        let mut empty = 0u8;
        for file in 0..8 {
            let code = piece(board, rank * 8 + file);
            if code == 0 {
                empty += 1;
                continue;
            }
            if empty > 0 {
                text.push(char::from(b'0' + empty));
                empty = 0;
            }
            let symbol = b" PNBRQKpnbrqk"
                .get(code as usize)
                .ok_or_else(|| io::Error::other("invalid piece code"))?;
            text.push(char::from(*symbol));
        }
        if empty > 0 {
            text.push(char::from(b'0' + empty));
        }
        if rank > 0 {
            text.push('/');
        }
    }
    text.push_str(if black { " b " } else { " w " });
    if board.castling > 15 || board.ep_square > 64 {
        return Err(io::Error::other("invalid castling or en passant encoding"));
    }
    if board.castling == 0 {
        text.push('-');
    } else {
        for (bit, symbol) in "KQkq".chars().enumerate() {
            if board.castling & (1 << bit) != 0 {
                text.push(symbol);
            }
        }
    }
    text.push(' ');
    if board.ep_square == 64 {
        text.push('-');
    } else {
        text.push(char::from(b'a' + board.ep_square % 8));
        text.push(char::from(b'1' + board.ep_square / 8));
    }
    text.push_str(&format!(" {} 1", board.halfmove_clock));
    CString::new(text).map_err(io::Error::other)
}

#[cfg(test)]
mod tests;

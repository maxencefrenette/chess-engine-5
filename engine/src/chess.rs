use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

impl Color {
    fn other(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }

    fn pawn_step(self) -> i8 {
        if self == Self::White { 1 } else { -1 }
    }

    fn back_rank(self) -> u8 {
        if self == Self::White { 0 } else { 7 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Piece {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

use Piece::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move {
    from: u8,
    to: u8,
    promotion: Option<Piece>,
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for square in [self.from, self.to] {
            write!(f, "{}{}", char::from(b'a' + square % 8), 1 + square / 8)?;
        }
        if let Some(piece) = self.promotion {
            let suffix = match piece {
                Queen => 'q',
                Rook => 'r',
                Bishop => 'b',
                Knight => 'n',
                _ => unreachable!("invalid promotion"),
            };
            write!(f, "{suffix}")?;
        }
        Ok(())
    }
}

const KNIGHT_STEPS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];
const ORTHOGONAL: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const DIAGONAL: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

fn offset(square: u8, file: i8, rank: i8) -> Option<u8> {
    let file = (square % 8) as i8 + file;
    let rank = (square / 8) as i8 + rank;
    if (0..8).contains(&file) && (0..8).contains(&rank) {
        Some((rank * 8 + file) as u8)
    } else {
        None
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Position {
    board: [Option<(Color, Piece)>; 64],
    turn: Color,
    // Each corner bit records whether castling with that rook is still available.
    castling: u64,
    en_passant: Option<u8>,
}

#[derive(Clone, Debug)]
pub struct Game {
    position: Position,
    history: Vec<Position>,
    halfmoves: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Win(Color),
    Draw,
}

impl Default for Game {
    fn default() -> Self {
        let mut board = [None; 64];
        let back = [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook];
        for (file, piece) in back.into_iter().enumerate() {
            board[file] = Some((Color::White, piece));
            board[8 + file] = Some((Color::White, Pawn));
            board[48 + file] = Some((Color::Black, Pawn));
            board[56 + file] = Some((Color::Black, piece));
        }
        Self {
            position: Position {
                board,
                turn: Color::White,
                castling: 1 | (1 << 7) | (1 << 56) | (1 << 63),
                en_passant: None,
            },
            history: Vec::new(),
            halfmoves: 0,
        }
    }
}

impl Game {
    pub fn turn(&self) -> Color {
        self.position.turn
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        self.position.legal_moves()
    }

    pub fn play(&mut self, mv: Move) -> bool {
        let moves = self.legal_moves();
        if self.outcome_with_moves(&moves).is_some() || !moves.contains(&mv) {
            return false;
        }
        self.play_legal(mv);
        true
    }

    pub fn outcome(&self) -> Option<Outcome> {
        self.outcome_with_moves(&self.legal_moves())
    }

    pub(crate) fn outcome_with_moves(&self, moves: &[Move]) -> Option<Outcome> {
        if moves.is_empty() {
            return Some(if self.position.in_check(self.turn()) {
                Outcome::Win(self.turn().other())
            } else {
                Outcome::Draw
            });
        }
        // For self-play, always claim a fifty-move or threefold draw when available.
        if self.halfmoves >= 100 || self.position.insufficient_material() {
            return Some(Outcome::Draw);
        }
        if self.history.len() >= 8 {
            let key = self.position.repetition_key(moves);
            if self
                .history
                .iter()
                .rev()
                .filter(|prior| **prior == key)
                .nth(1)
                .is_some()
            {
                return Some(Outcome::Draw);
            }
        }
        None
    }

    pub(crate) fn play_legal(&mut self, mv: Move) {
        let (_, piece) = self.position.board[mv.from as usize].unwrap();
        let capture = self.position.board[mv.to as usize].is_some();
        let next = self.position.after(mv);
        if piece == Pawn || capture || next.castling != self.position.castling {
            self.history.clear();
        } else {
            let mut key = self.position.clone();
            // En passant only changes position identity if the capture is legal.
            if key.en_passant.is_some() {
                key = key.repetition_key(&key.legal_moves());
            }
            self.history.push(key);
        }
        if piece == Pawn || capture {
            self.halfmoves = 0;
        } else {
            self.halfmoves += 1;
        }
        self.position = next;
    }

    /// Material balance in pawn units, from the side to move.
    pub fn material_balance(&self) -> i32 {
        self.position
            .board
            .iter()
            .flatten()
            .map(|&(color, piece)| {
                let value = match piece {
                    Pawn => 1,
                    Knight | Bishop => 3,
                    Rook => 5,
                    Queen => 9,
                    King => 0,
                };
                if color == self.turn() { value } else { -value }
            })
            .sum()
    }
}

impl Position {
    fn legal_moves(&self) -> Vec<Move> {
        let mut moves = Vec::with_capacity(48);
        for from in 0..64u8 {
            let Some((color, piece)) = self.board[from as usize] else {
                continue;
            };
            if color != self.turn {
                continue;
            }
            match piece {
                Pawn => self.pawn_moves(from, &mut moves),
                Knight => self.ray_moves(from, &KNIGHT_STEPS, 1, &mut moves),
                Bishop => self.ray_moves(from, &DIAGONAL, 7, &mut moves),
                Rook => self.ray_moves(from, &ORTHOGONAL, 7, &mut moves),
                Queen | King => {
                    let distance = if piece == King { 1 } else { 7 };
                    self.ray_moves(from, &DIAGONAL, distance, &mut moves);
                    self.ray_moves(from, &ORTHOGONAL, distance, &mut moves);
                    if piece == King {
                        self.castling_moves(from, &mut moves);
                    }
                }
            }
        }
        moves.retain(|&mv| !self.after(mv).in_check(self.turn));
        moves
    }

    fn pawn_moves(&self, from: u8, moves: &mut Vec<Move>) {
        let step = self.turn.pawn_step();
        let Some(ahead) = offset(from, 0, step) else {
            return;
        };
        if self.board[ahead as usize].is_none() {
            self.push_pawn_move(from, ahead, moves);
            let home = if self.turn == Color::White { 1 } else { 6 };
            if from / 8 == home {
                let double = offset(from, 0, step * 2).unwrap();
                if self.board[double as usize].is_none() {
                    self.push_pawn_move(from, double, moves);
                }
            }
        }
        for file in [-1, 1] {
            if let Some(to) = offset(from, file, step) {
                let capture = self.board[to as usize].is_some_and(|(color, _)| color != self.turn);
                if capture || self.en_passant == Some(to) {
                    self.push_pawn_move(from, to, moves);
                }
            }
        }
    }

    fn push_pawn_move(&self, from: u8, to: u8, moves: &mut Vec<Move>) {
        if to / 8 == self.turn.other().back_rank() {
            moves.extend([Queen, Rook, Bishop, Knight].map(|piece| Move {
                from,
                to,
                promotion: Some(piece),
            }));
        } else {
            moves.push(Move {
                from,
                to,
                promotion: None,
            });
        }
    }

    fn ray_moves(&self, from: u8, steps: &[(i8, i8)], distance: i8, moves: &mut Vec<Move>) {
        for &(file, rank) in steps {
            for step in 1..=distance {
                let Some(to) = offset(from, file * step, rank * step) else {
                    break;
                };
                let occupant = self.board[to as usize];
                if !occupant.is_some_and(|(color, _)| color == self.turn) {
                    moves.push(Move {
                        from,
                        to,
                        promotion: None,
                    });
                }
                if occupant.is_some() {
                    break;
                }
            }
        }
    }

    fn castling_moves(&self, from: u8, moves: &mut Vec<Move>) {
        let base = 8 * self.turn.back_rank();
        if from != base + 4 || self.castling & (0xff << base) == 0 || self.in_check(self.turn) {
            return;
        }
        for (rook, transit, to, empty) in [(7, 5, 6, 5..7), (0, 3, 2, 1..4)] {
            if self.castling & (1 << (base + rook)) == 0
                || empty
                    .into_iter()
                    .any(|file| self.board[(base + file) as usize].is_some())
            {
                continue;
            }
            let across = Move {
                from,
                to: base + transit,
                promotion: None,
            };
            if !self.after(across).in_check(self.turn) {
                moves.push(Move {
                    from,
                    to: base + to,
                    promotion: None,
                });
            }
        }
    }

    fn after(&self, mv: Move) -> Self {
        let mut next = self.clone();
        let (color, piece) = next.board[mv.from as usize].take().unwrap();
        next.board[mv.to as usize] = Some((color, mv.promotion.unwrap_or(piece)));
        next.castling &= !((1 << mv.from) | (1 << mv.to));
        next.en_passant = None;
        next.turn = color.other();

        if piece == Pawn {
            if self.en_passant == Some(mv.to) {
                let captured = offset(mv.to, 0, -color.pawn_step()).unwrap();
                next.board[captured as usize] = None;
            }
            if mv.from.abs_diff(mv.to) == 16 {
                next.en_passant = Some((mv.from + mv.to) / 2);
            }
        }
        if piece == King {
            let base = 8 * color.back_rank();
            next.castling &= !(0xff << base);
            if mv.from.abs_diff(mv.to) == 2 {
                let (from, to) = if mv.to > mv.from { (7, 5) } else { (0, 3) };
                next.board[(base + to) as usize] = next.board[(base + from) as usize].take();
            }
        }
        next
    }

    fn in_check(&self, color: Color) -> bool {
        let king = self
            .board
            .iter()
            .position(|&piece| piece == Some((color, King)))
            .expect("a legal position has both kings") as u8;
        let enemy = color.other();
        for file in [-1, 1] {
            if offset(king, file, color.pawn_step())
                .is_some_and(|square| self.board[square as usize] == Some((enemy, Pawn)))
            {
                return true;
            }
        }
        for (file, rank) in KNIGHT_STEPS {
            if offset(king, file, rank)
                .is_some_and(|square| self.board[square as usize] == Some((enemy, Knight)))
            {
                return true;
            }
        }
        for (steps, slider) in [(ORTHOGONAL, Rook), (DIAGONAL, Bishop)] {
            for (file, rank) in steps {
                for distance in 1..=7 {
                    let Some(square) = offset(king, file * distance, rank * distance) else {
                        break;
                    };
                    if let Some((attacker, piece)) = self.board[square as usize] {
                        if attacker == enemy
                            && (piece == slider
                                || piece == Queen
                                || (piece == King && distance == 1))
                        {
                            return true;
                        }
                        break;
                    }
                }
            }
        }
        false
    }

    fn repetition_key(&self, moves: &[Move]) -> Self {
        let mut key = self.clone();
        if let Some(target) = self.en_passant {
            let can_capture = moves.iter().any(|mv| {
                mv.to == target && self.board[mv.from as usize] == Some((self.turn, Pawn))
            });
            if !can_capture {
                key.en_passant = None;
            }
        }
        key
    }

    fn insufficient_material(&self) -> bool {
        let mut knights = 0;
        let mut bishop_color = None;
        for (square, occupant) in self.board.iter().enumerate() {
            match occupant {
                None | Some((_, King)) => {}
                Some((_, Knight)) => knights += 1,
                Some((_, Bishop)) => {
                    let color = (square / 8 + square % 8) % 2;
                    if bishop_color.is_some_and(|previous| previous != color) {
                        return false;
                    }
                    bishop_color = Some(color);
                }
                _ => return false,
            }
        }
        knights == 0 || (knights == 1 && bishop_color.is_none())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perft(position: &Position, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let moves = position.legal_moves();
        if depth == 1 {
            return moves.len() as u64;
        }
        moves
            .into_iter()
            .map(|mv| perft(&position.after(mv), depth - 1))
            .sum()
    }

    #[test]
    fn start_position_perft() {
        assert_eq!(perft(&Game::default().position, 4), 197_281);
    }

    #[test]
    fn tactical_position_perft() {
        // Kiwipete includes castling, pins, and en passant in its first three plies.
        let ranks = [
            "r...k..r", "p.ppqpb.", "bn..pnp.", "...PN...", ".p..P...", "..N..Q.p", "PPPBBPPP",
            "R...K..R",
        ];
        let mut position = Game::default().position;
        for (rank, row) in ranks.into_iter().rev().enumerate() {
            for (file, symbol) in row.chars().enumerate() {
                let piece = match symbol.to_ascii_lowercase() {
                    'p' => Pawn,
                    'n' => Knight,
                    'b' => Bishop,
                    'r' => Rook,
                    'q' => Queen,
                    'k' => King,
                    _ => {
                        position.board[8 * rank + file] = None;
                        continue;
                    }
                };
                let color = if symbol.is_ascii_uppercase() {
                    Color::White
                } else {
                    Color::Black
                };
                position.board[8 * rank + file] = Some((color, piece));
            }
        }
        assert_eq!(perft(&position, 3), 97_862);
    }

    #[test]
    fn repetition_ignores_en_passant_without_a_legal_capture() {
        let mut game = Game::default();
        for name in [
            "a2a4", "g8f6", "g1f3", "f6g8", "f3g1", "g8f6", "g1f3", "f6g8", "f3g1",
        ] {
            assert_eq!(game.outcome(), None);
            let mv = game
                .legal_moves()
                .into_iter()
                .find(|mv| mv.to_string() == name)
                .unwrap();
            assert!(game.play(mv));
        }
        assert_eq!(game.outcome(), Some(Outcome::Draw));
    }
}

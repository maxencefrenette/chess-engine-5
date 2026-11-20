//! Map Black's positions and packed move IDs into White's coordinate frame.
use chess_data::Board;

#[derive(Clone, Copy, Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Perspective {
    Absolute,
    SideToMove,
}

impl Perspective {
    /// Return the network board, its side to move, and the XOR mask for packed moves.
    pub fn orient(self, board: Board, black: bool) -> (Board, bool, u16) {
        let mask = self.move_mask(black);
        if mask != 0 {
            (flip_board(&board), false, mask)
        } else {
            (board, black, 0)
        }
    }

    pub fn move_mask(self, black: bool) -> u16 {
        if black && matches!(self, Self::SideToMove) {
            56 | (56 << 6)
        } else {
            0
        }
    }
}

const fn swap_piece(piece: u8) -> u8 {
    match piece {
        1..=6 => piece + 6,
        7..=12 => piece - 6,
        _ => piece,
    }
}

const SWAP_COLORS: [u8; 256] = {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = swap_piece(i as u8 & 15) | (swap_piece(i as u8 >> 4) << 4);
        i += 1;
    }
    table
};

pub(crate) fn flip_board(board: &Board) -> Board {
    Board {
        board: std::array::from_fn(|i| SWAP_COLORS[board.board[i ^ 28] as usize]),
        castling: ((board.castling & 3) << 2) | (board.castling >> 2),
        ep_square: if board.ep_square < 64 {
            board.ep_square ^ 56
        } else {
            board.ep_square
        },
        halfmove_clock: board.halfmove_clock,
        repetitions: board.repetitions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn black_castling_en_passant_and_promotions_have_white_coordinates() {
        let mut board = Board {
            board: [0; 32],
            castling: 6,
            ep_square: 18,
            halfmove_clock: 1,
            repetitions: 2,
        };
        board.board[30] = 12; // black king on e8
        board.board[4] = 7; // black pawn on a2
        let white = flip_board(&board);
        assert_eq!(white.board[2], 6);
        assert_eq!(white.board[24], 1);
        assert_eq!(white.castling, 9);
        assert_eq!(white.ep_square, 42);
        assert_eq!(flip_board(&white), board);
        let mask = Perspective::SideToMove.move_mask(true);
        for (black, white) in [
            (60 | 62 << 6, 4 | 6 << 6),
            (25 | 18 << 6, 33 | 42 << 6),
            (8 | 4 << 12, 48 | 56 << 6 | 4 << 12),
        ] {
            assert_eq!(black ^ mask, white);
        }
    }
}

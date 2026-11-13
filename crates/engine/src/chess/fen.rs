use super::*;

impl Game {
    /// Load orthodox FEN or the four position fields of an EPD record.
    pub fn from_fen(text: &str) -> Result<Self, &'static str> {
        let fields: Vec<_> = text.split_whitespace().collect();
        if fields.len() != 4 && fields.len() != 6 {
            return Err("expected four EPD or six FEN fields");
        }
        let ranks: Vec<_> = fields[0].split('/').collect();
        if ranks.len() != 8 {
            return Err("expected eight ranks");
        }
        let mut board = [None; 64];
        for (rank, row) in ranks.into_iter().rev().enumerate() {
            let mut file = 0;
            for symbol in row.chars() {
                if ('1'..='8').contains(&symbol) {
                    file += symbol as usize - '0' as usize;
                } else {
                    if file >= 8 {
                        return Err("rank exceeds eight squares");
                    }
                    let piece = match symbol.to_ascii_lowercase() {
                        'p' => Pawn,
                        'n' => Knight,
                        'b' => Bishop,
                        'r' => Rook,
                        'q' => Queen,
                        'k' => King,
                        _ => return Err("invalid piece"),
                    };
                    if piece == Pawn && (rank == 0 || rank == 7) {
                        return Err("pawn on back rank");
                    }
                    let color = if symbol.is_ascii_uppercase() {
                        Color::White
                    } else {
                        Color::Black
                    };
                    board[rank * 8 + file] = Some((color, piece));
                    file += 1;
                }
            }
            if file != 8 {
                return Err("rank must contain eight squares");
            }
        }
        for color in [Color::White, Color::Black] {
            if board.iter().filter(|&&p| p == Some((color, King))).count() != 1 {
                return Err("expected one king per side");
            }
        }
        let turn = match fields[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err("invalid side to move"),
        };
        let mut castling = 0;
        if fields[2] != "-" {
            for flag in fields[2].chars() {
                let (rook, king, color) = match flag {
                    'K' => (7, 4, Color::White),
                    'Q' => (0, 4, Color::White),
                    'k' => (63, 60, Color::Black),
                    'q' => (56, 60, Color::Black),
                    _ => return Err("invalid orthodox castling rights"),
                };
                if castling & (1_u64 << rook) != 0
                    || board[rook] != Some((color, Rook))
                    || board[king] != Some((color, King))
                {
                    return Err("castling rights do not match the board");
                }
                castling |= 1_u64 << rook;
            }
        }
        let en_passant = if fields[3] == "-" {
            None
        } else {
            let bytes = fields[3].as_bytes();
            let rank = if turn == Color::White { b'6' } else { b'3' };
            if bytes.len() != 2 || !(b'a'..=b'h').contains(&bytes[0]) || bytes[1] != rank {
                return Err("invalid en passant square");
            }
            let square = (rank - b'1') * 8 + bytes[0] - b'a';
            let pawn = offset(square, 0, -turn.pawn_step()).unwrap();
            if board[square as usize].is_some()
                || board[pawn as usize] != Some((turn.other(), Pawn))
            {
                return Err("en passant square does not match the board");
            }
            Some(square)
        };
        let halfmoves = if fields.len() == 6 {
            if fields[5]
                .parse::<u32>()
                .map_err(|_| "invalid fullmove number")?
                == 0
            {
                return Err("fullmove number must be positive");
            }
            fields[4].parse().map_err(|_| "invalid halfmove clock")?
        } else {
            0
        };
        let position = Position {
            board,
            turn,
            castling,
            en_passant,
        };
        if position.in_check(turn.other()) {
            return Err("side that just moved is in check");
        }
        Ok(Self {
            position,
            history: Vec::new(),
            halfmoves,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fen_preserves_turn_castling_en_passant_and_clock() {
        let mut game = Game::from_fen("r3k2r/8/8/8/3pP3/8/8/R3K2R b KQkq e3 0 15").unwrap();
        assert_eq!(game.turn(), Color::Black);
        assert_eq!(game.position.castling, Game::default().position.castling);
        assert_eq!(game.position.en_passant, Some(20));
        let moves = game.legal_moves();
        assert!(moves.iter().any(|m| m.to_string() == "e8g8"));
        assert!(moves.iter().any(|m| m.to_string() == "e8c8"));
        let capture = *moves.iter().find(|m| m.to_string() == "d4e3").unwrap();
        assert!(game.play(capture));
        assert_eq!(game.position.board[28], None);
        assert_eq!(game.position.board[20], Some((Color::Black, Pawn)));
        let game = Game::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 100 60").unwrap();
        assert_eq!(game.outcome(), Some(Outcome::Draw));
    }

    #[test]
    fn rejects_malformed_positions() {
        for fen in [
            "",
            "8/8 w - -",
            "9/8/8/8/8/8/8/8 w - -",
            "8/8/8/8/8/8/8/8 w - -",
            "4k3/8/8/8/8/8/8/4K3 x - -",
            "4k3/8/8/8/8/8/8/4K3 w K -",
            "4k3/8/8/8/8/8/8/4K3 w - e3",
            "4k3/8/8/8/8/8/8/4K3 w - - 0 0",
        ] {
            assert!(Game::from_fen(fen).is_err(), "{fen}");
        }
    }
}

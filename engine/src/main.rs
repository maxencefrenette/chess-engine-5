use std::{env, num::NonZeroU32, process::ExitCode};

use chess_engine::{chess::Game, mcts::search};

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let simulations = args
        .next()
        .ok_or("usage: chess-engine SIMULATIONS [MOVE ...]")?
        .parse::<NonZeroU32>()
        .map_err(|_| "SIMULATIONS must be a positive integer")?;
    let mut game = Game::default();

    for name in args {
        let mv = game
            .legal_moves()
            .into_iter()
            .find(|mv| mv.to_string() == name)
            .ok_or_else(|| format!("illegal move: {name}"))?;
        if !game.play(mv) {
            return Err(format!("game already ended before {name}"));
        }
    }

    if let Some(outcome) = game.outcome() {
        println!("{outcome:?}");
        return Ok(());
    }

    let mut moves = search(&game, simulations, |game| {
        let material = game.material_balance() as f32;
        // Keep heuristic evaluations strictly inside the range of game results.
        material / (material.abs() + 8.0)
    });
    moves.sort_by_key(|entry| std::cmp::Reverse(entry.visits));
    for entry in moves {
        println!("{} {}", entry.mv, entry.visits);
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

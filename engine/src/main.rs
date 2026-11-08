use std::{
    env,
    num::{NonZeroU32, NonZeroUsize},
    path::Path,
    process::ExitCode,
};

use chess_engine::{
    chess::Game,
    mcts::{material_value, search},
    self_play,
};

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let first = args
        .next()
        .ok_or("usage: chess-engine SIMULATIONS [MOVE ...] | self-play OUTPUT_DIR GAMES SEED")?;
    if first == "self-play" {
        let (Some(output), Some(games), Some(seed), None) =
            (args.next(), args.next(), args.next(), args.next())
        else {
            return Err("usage: chess-engine self-play OUTPUT_DIR GAMES SEED".into());
        };
        let games = games
            .parse::<NonZeroUsize>()
            .map_err(|_| "GAMES must be a positive integer")?;
        let seed = seed
            .parse::<u64>()
            .map_err(|_| "SEED must be an unsigned 64-bit integer")?;
        let run =
            self_play::generate(Path::new(&output), games, seed).map_err(|err| err.to_string())?;
        println!("{}", run.display());
        return Ok(());
    }
    let simulations = first
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

    let mut moves = search(&game, simulations, material_value);
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

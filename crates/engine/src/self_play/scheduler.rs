use std::{io, ops::Range, sync::Mutex};

use chess_network::inference::Prediction;

use super::{ActiveGame, Config, PlayedGame};
use crate::{mcts::Leaf, model_evaluator::Input};

struct Shard<'a> {
    config: &'a Config,
    remaining: Mutex<Range<usize>>,
}

#[derive(Default)]
struct Group {
    active: Vec<ActiveGame>,
    leaves: Vec<(usize, Leaf)>,
    finished: Vec<(usize, i8)>,
    completed: Vec<(usize, PlayedGame)>,
}

impl Group {
    fn advance(&mut self, predictions: Vec<Prediction>, shard: &Shard) -> Option<Vec<Input>> {
        for ((i, leaf), result) in self.leaves.drain(..).zip(predictions) {
            self.active[i].search.evaluate(
                leaf,
                result.value,
                result.draw,
                &result.priors,
                shard.config.search.policy_temperature,
            );
        }
        loop {
            // Replace slots only after all replies referring to those indices were applied.
            for (i, white_value) in self.finished.drain(..).rev() {
                let next = shard.remaining.lock().unwrap().next();
                let game = match next {
                    Some(id) => std::mem::replace(
                        &mut self.active[i],
                        ActiveGame::new(shard.config.seed, id),
                    ),
                    None => self.active.swap_remove(i),
                };
                self.completed.push((
                    game.id,
                    PlayedGame {
                        positions: game.positions,
                        white_value,
                    },
                ));
            }
            if self.active.is_empty() {
                return None;
            }
            for (i, game) in self.active.iter_mut().enumerate() {
                let outcome = game.collect(shard.config, |leaf| self.leaves.push((i, leaf)));
                if let Some(value) = outcome {
                    self.finished.push((i, value));
                }
            }
            if !self.leaves.is_empty() {
                return Some(self.leaves.iter().map(|(_, leaf)| leaf.input()).collect());
            }
        }
    }
}

pub(super) fn play_games(
    config: &Config,
    ids: Range<usize>,
    mut evaluate: impl FnMut(Vec<Input>) -> io::Result<Vec<Prediction>>,
) -> io::Result<Vec<(usize, PlayedGame)>> {
    let slots = config.batch.get().min(ids.len());
    let shard = Shard {
        config,
        remaining: Mutex::new(ids),
    };
    let groups = crate::inference_queue::run(
        slots,
        config.search_threads,
        |n| Group {
            active: shard
                .remaining
                .lock()
                .unwrap()
                .by_ref()
                .take(n)
                .map(|id| ActiveGame::new(config.seed, id))
                .collect(),
            ..Group::default()
        },
        |group, predictions| group.advance(predictions, &shard).map(|inputs| (0, inputs)),
        |_, inputs| evaluate(inputs).map(Some),
    )?;
    let mut games: Vec<_> = groups
        .into_iter()
        .flat_map(|group| group.completed)
        .collect();
    games.sort_unstable_by_key(|(id, _)| *id);
    Ok(games)
}

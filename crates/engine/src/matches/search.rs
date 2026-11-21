use super::{ActiveGame, Outcome};
use crate::{chess::Game, mcts::Leaf, model_evaluator::Input};
use chess_network::inference::Prediction;
use std::{
    collections::BTreeMap,
    num::{NonZeroU16, NonZeroU32},
    ops::Range,
    sync::Mutex,
};

pub(super) struct Work {
    pub ids: Range<u64>,
    pub results: BTreeMap<u64, usize>,
}

impl Work {
    // Keep stopping decisions independent of game completion order.
    pub fn take_pair(&mut self, pair: u64) -> Option<[usize; 2]> {
        let first = 2 * pair;
        self.results.get(&first)?;
        let second = self.results.remove(&(first + 1))?;
        Some([self.results.remove(&first).unwrap(), second])
    }
}

pub(super) struct Group {
    games: Vec<Option<ActiveGame>>,
    pending: Vec<(usize, Leaf)>,
}

impl Group {
    pub fn new(n: usize, work: &Mutex<Work>, openings: &[Game]) -> Self {
        let games = work
            .lock()
            .unwrap()
            .ids
            .by_ref()
            .take(n)
            .map(|id| Some(ActiveGame::new(id, openings)))
            .collect();
        Self {
            games,
            pending: Vec::new(),
        }
    }

    pub fn advance(
        &mut self,
        predictions: Vec<Prediction>,
        work: &Mutex<Work>,
        openings: &[Game],
        visits: [NonZeroU32; 2],
        exploration: f32,
        leaves_per_game: NonZeroU16,
    ) -> Option<(usize, Vec<Input>)> {
        for ((i, leaf), result) in self.pending.drain(..).zip(predictions) {
            self.games[i].as_mut().unwrap().search.evaluate(
                leaf,
                result.value,
                result.draw,
                &result.priors,
            );
        }
        // Following the first active game keeps more of the group on the same checkpoint.
        while let Some(first) = self.games.iter().flatten().next() {
            let side = first.player();
            for (i, slot) in self.games.iter_mut().enumerate() {
                let Some(game) = slot else { continue };
                if game.player() != side {
                    continue;
                }
                let count = game.search.collect_leaves(
                    &game.game,
                    visits[side],
                    exploration,
                    leaves_per_game.get().into(),
                    |leaf| self.pending.push((i, leaf)),
                );
                if count == 0
                    && let Some(outcome) = game.play()
                {
                    let points = match outcome {
                        Outcome::Win(color) if color == game.candidate_color() => 2,
                        Outcome::Win(_) => 0,
                        Outcome::Draw => 1,
                    };
                    let mut work = work.lock().unwrap();
                    work.results.insert(game.id, points);
                    *slot = work.ids.next().map(|id| ActiveGame::new(id, openings));
                }
            }
            if !self.pending.is_empty() {
                let inputs = self.pending.iter().map(|(_, l)| l.input()).collect();
                return Some((side, inputs));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_waits_for_both_games_of_the_next_pair() {
        let mut work = Work {
            ids: 0..4,
            results: [(3, 1), (2, 0), (1, 2)].into(),
        };
        assert_eq!(work.take_pair(0), None);
        work.results.insert(0, 1);
        assert_eq!(work.take_pair(0), Some([1, 2]));
        assert_eq!(work.take_pair(1), Some([0, 1]));
        assert!(work.results.is_empty());
    }

    #[test]
    fn each_color_uses_its_checkpoints_visit_budget() {
        let openings = [Game::default()];
        let visits = [2, 7].map(|n| NonZeroU32::new(n).unwrap());
        for id in 0..2 {
            let work = Mutex::new(Work {
                ids: id..id + 1,
                results: Default::default(),
            });
            let mut group = Group::new(1, &work, &openings);
            let mut predictions = Vec::new();
            let first = usize::from(id == 0);
            let second = 1 - first;
            // Each move requests the root prediction before spending its visits.
            for (expected, calls) in [
                (first, 1 + visits[first].get()),
                (second, 1 + visits[second].get()),
                (first, 1),
            ] {
                for _ in 0..calls {
                    let (side, inputs) = group
                        .advance(
                            std::mem::take(&mut predictions),
                            &work,
                            &openings,
                            visits,
                            1.4,
                            NonZeroU16::MIN,
                        )
                        .unwrap();
                    assert_eq!(side, expected);
                    assert_eq!(inputs.len(), 1);
                    predictions = inputs
                        .into_iter()
                        .map(|(_, moves)| Prediction {
                            value: 0.0,
                            draw: 0.5,
                            priors: vec![1.0 / moves.len() as f32; moves.len()],
                        })
                        .collect();
                }
            }
        }
    }
}

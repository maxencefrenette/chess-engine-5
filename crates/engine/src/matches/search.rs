use super::{ActiveGame, Outcome, Settings};
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
        settings: &[Settings; 2],
        leaves_per_game: NonZeroU16,
    ) -> Option<(usize, Vec<Input>)> {
        for ((i, leaf), result) in self.pending.drain(..).zip(predictions) {
            let game = self.games[i].as_mut().unwrap();
            let side = game.player();
            game.searches[side].evaluate(leaf, result.value, result.draw, &result.priors);
        }
        // Following the first active game keeps more of the group on the same checkpoint.
        while let Some(first) = self.games.iter().flatten().next() {
            let side = first.player();
            for (i, slot) in self.games.iter_mut().enumerate() {
                let Some(game) = slot else { continue };
                if game.player() != side {
                    continue;
                }
                let count = game.searches[side].collect_leaves(
                    &game.game,
                    visits[side],
                    settings[side].exploration,
                    leaves_per_game.get().into(),
                    |leaf| self.pending.push((i, leaf)),
                );
                if count == 0
                    && let Some(outcome) = game.play(settings)
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
    fn search_settings_and_reused_visits_follow_the_player_across_colors() {
        let openings = [Game::default()];
        let visits = [2, 7].map(|n| NonZeroU32::new(n).unwrap());
        let settings = [
            Settings {
                exploration: 0.0,
                reuse_subtree: false,
            },
            Settings {
                exploration: 1.4,
                reuse_subtree: true,
            },
        ];
        for id in 0..2 {
            let work = Mutex::new(Work {
                ids: id..id + 1,
                results: Default::default(),
            });
            let mut group = Group::new(1, &work, &openings);
            let mut predictions = Vec::new();
            let mut expected_game = openings[0].clone();
            let candidate_color = group.games[0].as_ref().unwrap().candidate_color();
            for ply in 0..5 {
                let expected = usize::from(expected_game.turn() == candidate_color);
                // Replaying the candidate's predicted line advances its tree two plies.
                let calls = if expected == 1 && ply >= 2 {
                    2
                } else {
                    1 + visits[expected].get()
                };
                for _ in 0..calls {
                    let (side, inputs) = group
                        .advance(
                            std::mem::take(&mut predictions),
                            &work,
                            &openings,
                            visits,
                            &settings,
                            NonZeroU16::MIN,
                        )
                        .unwrap();
                    assert_eq!(side, expected);
                    let actual = &group.games[0].as_ref().unwrap().game;
                    assert_eq!(
                        actual.snapshot(&actual.legal_moves()),
                        expected_game.snapshot(&expected_game.legal_moves())
                    );
                    assert_eq!(inputs.len(), 1);
                    predictions = inputs
                        .into_iter()
                        .map(|((_, black), moves)| {
                            let mut priors = vec![0.0; moves.len()];
                            let candidate_turn =
                                black == (candidate_color == crate::chess::Color::Black);
                            // The baseline ignores the last-move prior at zero exploration.
                            let preferred = if side == 0 || candidate_turn {
                                moves.len() - 1
                            } else {
                                0
                            };
                            priors[preferred] = 1.0;
                            Prediction {
                                value: 0.0,
                                draw: 0.5,
                                priors,
                            }
                        })
                        .collect();
                }
                let moves = expected_game.legal_moves();
                expected_game.play_legal(if expected == 1 {
                    *moves.last().unwrap()
                } else {
                    moves[0]
                });
            }
        }
    }
}

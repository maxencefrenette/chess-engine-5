use std::{num::NonZeroU32, ops::Range};

use crate::chess::{Game, Move, Outcome};

const EXPLORATION: f32 = 1.4;

#[derive(Clone, Copy, Debug)]
pub struct MoveVisits {
    pub mv: Move,
    pub visits: u32,
}

#[derive(Default)]
struct Node {
    visits: u32,
    prior: f32,
    value_sum: f32,
    state: NodeState,
}

#[derive(Default)]
enum NodeState {
    #[default]
    Unexpanded,
    Terminal(f32),
    Branch(Range<usize>),
}

#[derive(Default)]
struct Tree {
    nodes: Vec<Node>,
    moves: Vec<Move>,
}

impl Tree {
    fn expand(&mut self, index: usize, moves: Vec<Move>, priors: Option<&[f32]>) {
        let start = self.nodes.len();
        self.nodes.resize_with(start + moves.len(), Node::default);
        let uniform = 1. / moves.len() as f32;
        for (i, child) in self.nodes[start..].iter_mut().enumerate() {
            child.prior = priors.map_or(uniform, |priors| priors[i]);
        }
        self.moves.extend(moves);
        let children = start..self.nodes.len();
        self.nodes[index].state = NodeState::Branch(children);
    }

    fn select(&self, parent: usize, children: Range<usize>) -> usize {
        let exploration = (self.nodes[parent].visits.max(1) as f32).sqrt();
        children
            .max_by(|&left, &right| {
                self.score(left, exploration)
                    .total_cmp(&self.score(right, exploration))
                    .then_with(|| right.cmp(&left))
            })
            .unwrap()
    }

    fn score(&self, child: usize, exploration: f32) -> f32 {
        let node = &self.nodes[child];
        // Child values are from the opponent's perspective.
        let value = -node.value_sum / node.visits.max(1) as f32;
        value + EXPLORATION * node.prior * exploration / (1.0 + node.visits as f32)
    }
}

pub(crate) struct Leaf {
    pub game: Game,
    pub moves: Vec<Move>,
}

pub(crate) struct Search {
    tree: Tree,
    path: Vec<usize>,
}

impl Search {
    pub fn new(moves: Vec<Move>) -> Self {
        let mut search = Self::with_policy();
        search.tree.expand(0, moves, None);
        search
    }

    pub fn with_policy() -> Self {
        Self {
            tree: Tree {
                nodes: vec![Node::default()],
                moves: Vec::new(),
            },
            path: Vec::new(),
        }
    }

    /// Advance through terminal simulations until there is an evaluation to do.
    pub fn next_leaf(&mut self, game: &Game, budget: NonZeroU32) -> Option<Leaf> {
        while self.tree.nodes[0].visits < budget.get() {
            self.path.clear();
            let mut position = game.clone();
            let mut index = 0;
            let value = loop {
                self.path.push(index);
                match &self.tree.nodes[index].state {
                    NodeState::Branch(children) => {
                        index = self.tree.select(index, children.clone());
                        position.play_legal(self.tree.moves[index - 1]);
                    }
                    NodeState::Terminal(value) => break *value,
                    NodeState::Unexpanded => {
                        let moves = position.legal_moves();
                        if let Some(outcome) = position.outcome_with_moves(&moves) {
                            let value = match outcome {
                                Outcome::Draw => 0.0,
                                Outcome::Win(color) if color == position.turn() => 1.0,
                                Outcome::Win(_) => -1.0,
                            };
                            self.tree.nodes[index].state = NodeState::Terminal(value);
                            break value;
                        }
                        return Some(Leaf {
                            game: position,
                            moves,
                        });
                    }
                }
            };
            self.backup(value);
        }
        None
    }

    pub fn evaluate(&mut self, moves: Vec<Move>, value: f32, priors: Option<&[f32]>) {
        let index = *self.path.last().expect("a leaf is pending");
        self.tree.expand(index, moves, priors);
        if index != 0 {
            self.backup(value);
        }
    }

    fn backup(&mut self, mut value: f32) {
        for &index in self.path.iter().rev() {
            self.tree.nodes[index].visits += 1;
            self.tree.nodes[index].value_sum += value;
            value = -value;
        }
    }

    pub fn visits(&self) -> impl Iterator<Item = MoveVisits> + '_ {
        let NodeState::Branch(children) = &self.tree.nodes[0].state else {
            unreachable!("root must be expanded");
        };
        children.clone().map(|index| MoveVisits {
            mv: self.tree.moves[index - 1],
            visits: self.tree.nodes[index].visits,
        })
    }
}

/// Run uniform-prior PUCT and return the root visit counts in legal move order.
/// The leaf evaluator returns a value in [-1, 1] for the side to move.
/// A finished game returns no moves.
pub fn search(
    game: &Game,
    simulations: NonZeroU32,
    evaluate: impl Fn(&Game) -> f32,
) -> Vec<MoveVisits> {
    let moves = game.legal_moves();
    if game.outcome_with_moves(&moves).is_some() {
        return Vec::new();
    }

    let mut search = Search::new(moves);
    while let Some(leaf) = search.next_leaf(game, simulations) {
        search.evaluate(leaf.moves, evaluate(&leaf.game), None);
    }
    search.visits().collect()
}

/// Material-only leaf evaluation from the side to move.
pub fn material_value(game: &Game) -> f32 {
    let material = game.material_balance() as f32;
    // Keep heuristic evaluations strictly inside the range of game results.
    material / (material.abs() + 8.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_root_prior_guides_the_first_visit_without_using_the_budget() {
        let game = Game::default();
        let budget = NonZeroU32::new(1).unwrap();
        let mut search = Search::with_policy();
        let root = search.next_leaf(&game, budget).unwrap();
        let mut priors = vec![0.; root.moves.len()];
        *priors.last_mut().unwrap() = 1.;
        let expected = *root.moves.last().unwrap();
        search.evaluate(root.moves, 0., Some(&priors));
        let child = search.next_leaf(&game, budget).unwrap();
        search.evaluate(child.moves, 0., None);
        assert!(search.next_leaf(&game, budget).is_none());
        let visited: Vec<_> = search.visits().filter(|entry| entry.visits != 0).collect();
        assert_eq!(visited.len(), 1);
        assert_eq!(visited[0].mv, expected);
        assert_eq!(visited[0].visits, 1);
    }

    #[test]
    fn search_prefers_mate_with_a_neutral_evaluator() {
        let mut game = Game::default();
        for name in ["f2f3", "e7e5", "g2g4"] {
            let mv = game
                .legal_moves()
                .into_iter()
                .find(|mv| mv.to_string() == name)
                .unwrap();
            assert!(game.play(mv));
        }

        let budget = NonZeroU32::new(128).unwrap();
        let visits = search(&game, budget, |_| 0.0);
        assert_eq!(
            visits.iter().map(|entry| entry.visits).sum::<u32>(),
            budget.get()
        );
        let best = visits.into_iter().max_by_key(|entry| entry.visits).unwrap();
        assert_eq!(best.mv.to_string(), "d8h4");
    }
}

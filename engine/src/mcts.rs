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
    fn expand(&mut self, index: usize, moves: Vec<Move>) -> Range<usize> {
        let start = self.nodes.len();
        self.nodes.resize_with(start + moves.len(), Node::default);
        self.moves.extend(moves);
        let children = start..self.nodes.len();
        self.nodes[index].state = NodeState::Branch(children.clone());
        children
    }

    fn select(&self, parent: usize, children: Range<usize>) -> usize {
        let prior = 1.0 / children.len() as f32;
        let exploration = EXPLORATION * prior * (self.nodes[parent].visits.max(1) as f32).sqrt();
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
        value + exploration / (1.0 + node.visits as f32)
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

    let mut tree = Tree {
        nodes: vec![Node::default()],
        moves: Vec::new(),
    };
    let root_children = tree.expand(0, moves);
    let mut path = Vec::new();

    for _ in 0..simulations.get() {
        let mut position = game.clone();
        let mut index = 0;
        path.clear();
        let mut value = loop {
            path.push(index);
            match &tree.nodes[index].state {
                NodeState::Branch(children) => {
                    index = tree.select(index, children.clone());
                    position.play_legal(tree.moves[index - 1]);
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
                        tree.nodes[index].state = NodeState::Terminal(value);
                        break value;
                    }
                    tree.expand(index, moves);
                    break evaluate(&position);
                }
            }
        };

        for &index in path.iter().rev() {
            tree.nodes[index].visits += 1;
            tree.nodes[index].value_sum += value;
            value = -value;
        }
    }

    root_children
        .map(|index| MoveVisits {
            mv: tree.moves[index - 1],
            visits: tree.nodes[index].visits,
        })
        .collect()
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

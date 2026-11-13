use std::{num::NonZeroU32, ops::Range};

use crate::chess::{Game, Move, Outcome};

use rand::Rng;

#[derive(Clone, Copy, Debug)]
pub struct MoveVisits {
    pub mv: Move,
    pub visits: u32,
}

#[derive(Default)]
struct Node {
    visits: u32,
    pending: u32,
    prior: f32,
    value_sum: f32,
    state: NodeState,
}

#[derive(Default)]
enum NodeState {
    #[default]
    Unexpanded,
    Pending,
    Terminal(f32),
    Branch(Range<usize>),
}

#[derive(Default)]
struct Tree {
    nodes: Vec<Node>,
    moves: Vec<Move>,
}

impl Tree {
    fn expand(&mut self, index: usize, moves: Vec<Move>, priors: &[f32]) {
        let start = self.nodes.len();
        self.nodes.resize_with(start + moves.len(), Node::default);
        assert_eq!(moves.len(), priors.len());
        for (child, &prior) in self.nodes[start..].iter_mut().zip(priors) {
            child.prior = prior;
        }
        self.moves.extend(moves);
        let children = start..self.nodes.len();
        self.nodes[index].state = NodeState::Branch(children);
    }

    fn select(&self, parent: usize, children: Range<usize>, exploration: f32) -> Option<usize> {
        let parent = &self.nodes[parent];
        let parent_scale = ((parent.visits + parent.pending).max(1) as f32).sqrt();
        children
            .filter(|&child| !matches!(self.nodes[child].state, NodeState::Pending))
            .max_by(|&left, &right| {
                self.score(left, exploration, parent_scale)
                    .total_cmp(&self.score(right, exploration, parent_scale))
                    .then_with(|| right.cmp(&left))
            })
    }

    fn score(&self, child: usize, exploration: f32, parent_scale: f32) -> f32 {
        let node = &self.nodes[child];
        let visits = node.visits + node.pending;
        // Child values are from the opponent's perspective, so +1 is a virtual loss.
        let value = -(node.value_sum + node.pending as f32) / visits.max(1) as f32;
        value + exploration * node.prior * parent_scale / (1.0 + visits as f32)
    }

    fn backup(&mut self, path: &[usize], mut value: f32) {
        for &index in path.iter().rev() {
            self.nodes[index].visits += 1;
            self.nodes[index].value_sum += value;
            value = -value;
        }
    }
}

pub(crate) struct Leaf {
    pub game: Game,
    pub moves: Vec<Move>,
    path: Vec<usize>,
}

pub(crate) struct Search {
    tree: Tree,
    path: Vec<usize>,
}

impl Search {
    pub fn new() -> Self {
        Self {
            tree: Tree {
                nodes: vec![Node::default()],
                moves: Vec::new(),
            },
            path: Vec::new(),
        }
    }

    /// Advance through terminal simulations until there is an evaluation to do.
    pub fn next_leaf(&mut self, game: &Game, budget: NonZeroU32, exploration: f32) -> Option<Leaf> {
        while self.tree.nodes[0].visits + self.tree.nodes[0].pending < budget.get() {
            self.path.clear();
            let mut position = game.clone();
            let mut index = 0;
            let value = loop {
                self.path.push(index);
                match &self.tree.nodes[index].state {
                    NodeState::Branch(children) => {
                        index = self.tree.select(index, children.clone(), exploration)?;
                        position.play_legal(self.tree.moves[index - 1]);
                    }
                    NodeState::Terminal(value) => break *value,
                    NodeState::Pending => return None,
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
                        self.tree.nodes[index].state = NodeState::Pending;
                        for &node in &self.path {
                            self.tree.nodes[node].pending += 1;
                        }
                        return Some(Leaf {
                            game: position,
                            moves,
                            path: std::mem::take(&mut self.path),
                        });
                    }
                }
            };
            self.tree.backup(&self.path, value);
        }
        None
    }

    pub fn evaluate(&mut self, leaf: Leaf, value: f32, priors: &[f32]) {
        let index = *leaf.path.last().expect("a leaf has a path");
        for &node in &leaf.path {
            self.tree.nodes[node].pending -= 1;
        }
        self.tree.expand(index, leaf.moves, priors);
        if index != 0 {
            self.tree.backup(&leaf.path, value);
        }
        self.path = leaf.path;
    }

    /// Temperature-zero selection, with ties resolved by legal move order.
    pub fn best_move(&self) -> Move {
        self.visits()
            .rev()
            .max_by_key(|entry| entry.visits)
            .unwrap()
            .mv
    }

    pub fn choose_move(&self, temperature: f64, rng: &mut impl Rng) -> Move {
        if temperature == 1.0 {
            let mut ticket = rng.random_range(0..self.tree.nodes[0].visits);
            for entry in self.visits() {
                if ticket < entry.visits {
                    return entry.mv;
                }
                ticket -= entry.visits;
            }
            unreachable!("root visits sum to the search budget");
        }

        // Reverse the iteration so ties select the first legal move.
        let best = self
            .visits()
            .rev()
            .max_by_key(|entry| entry.visits)
            .unwrap();
        if temperature == 0.0 {
            return best.mv;
        }

        // Scaling by the maximum keeps at least one weight at 1, even at low temperatures.
        let weight = |visits| {
            if visits == best.visits {
                1.0
            } else {
                (f64::from(visits) / f64::from(best.visits)).powf(temperature.recip())
            }
        };
        let total: f64 = self.visits().map(|entry| weight(entry.visits)).sum();
        let mut ticket = rng.random_range(0.0..total);
        for entry in self.visits() {
            ticket -= weight(entry.visits);
            if ticket < 0.0 {
                return entry.mv;
            }
        }
        best.mv
    }

    pub fn visits(&self) -> impl DoubleEndedIterator<Item = MoveVisits> + '_ {
        let NodeState::Branch(children) = &self.tree.nodes[0].state else {
            unreachable!("root must be expanded");
        };
        children.clone().map(|index| MoveVisits {
            mv: self.tree.moves[index - 1],
            visits: self.tree.nodes[index].visits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_root_prior_guides_the_first_visit_without_using_the_budget() {
        let game = Game::default();
        let budget = NonZeroU32::new(1).unwrap();
        let mut search = Search::new();
        let root = search.next_leaf(&game, budget, 1.4).unwrap();
        let mut priors = vec![0.; root.moves.len()];
        *priors.last_mut().unwrap() = 1.;
        let expected = *root.moves.last().unwrap();
        search.evaluate(root, 0., &priors);
        let child = search.next_leaf(&game, budget, 1.4).unwrap();
        let priors = vec![1. / child.moves.len() as f32; child.moves.len()];
        search.evaluate(child, 0., &priors);
        assert!(search.next_leaf(&game, budget, 1.4).is_none());
        let visited: Vec<_> = search.visits().filter(|entry| entry.visits != 0).collect();
        assert_eq!(visited.len(), 1);
        assert_eq!(visited[0].mv, expected);
        assert_eq!(visited[0].visits, 1);
    }

    #[test]
    fn pending_leaves_are_distinct_and_reserve_the_visit_budget() {
        let game = Game::default();
        let budget = NonZeroU32::new(2).unwrap();
        let mut search = Search::new();
        let root = search.next_leaf(&game, budget, 1.4).unwrap();
        assert!(search.next_leaf(&game, budget, 1.4).is_none());
        let uniform = |leaf: &Leaf| vec![1.0 / leaf.moves.len() as f32; leaf.moves.len()];
        let priors = uniform(&root);
        search.evaluate(root, 0.0, &priors);

        let first = search.next_leaf(&game, budget, 1.4).unwrap();
        let second = search.next_leaf(&game, budget, 1.4).unwrap();
        assert_ne!(first.path, second.path);
        assert!(search.next_leaf(&game, budget, 1.4).is_none());
        assert_eq!(search.visits().map(|entry| entry.visits).sum::<u32>(), 0);
        for leaf in [second, first] {
            let priors = uniform(&leaf);
            search.evaluate(leaf, 0.25, &priors);
        }

        assert!(search.next_leaf(&game, budget, 1.4).is_none());
        assert_eq!(search.visits().filter(|entry| entry.visits == 1).count(), 2);
        assert!(search.tree.nodes.iter().all(|node| node.pending == 0));
        assert_eq!(search.tree.nodes[0].value_sum, -0.5);
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
        let mut search = Search::new();
        while let Some(leaf) = search.next_leaf(&game, budget, 1.4) {
            let priors = vec![1. / leaf.moves.len() as f32; leaf.moves.len()];
            search.evaluate(leaf, 0., &priors);
        }
        let visits: Vec<_> = search.visits().collect();
        assert_eq!(
            visits.iter().map(|entry| entry.visits).sum::<u32>(),
            budget.get()
        );
        let best = visits.into_iter().max_by_key(|entry| entry.visits).unwrap();
        assert_eq!(best.mv.to_string(), "d8h4");
    }
}

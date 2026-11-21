use std::{
    io,
    ops::Range,
    sync::{Mutex, mpsc},
    thread,
};

use chess_network::inference::Prediction;

use super::{ActiveGame, Config, PlayedGame, PlayoutCap, Settings};
use crate::{mcts::Leaf, model_evaluator::Input};

struct Shard<'a> {
    settings: &'a Settings,
    config: &'a Config,
    fast_visits: std::num::NonZeroU16,
    full_search_rate: f64,
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
            self.active[i]
                .search
                .evaluate(leaf, result.value, result.draw, &result.priors);
        }
        loop {
            // Replace slots only after all replies referring to those indices were applied.
            for (i, white_value) in self.finished.drain(..).rev() {
                let next = shard.remaining.lock().unwrap().next();
                let game = match next {
                    Some(id) => std::mem::replace(
                        &mut self.active[i],
                        ActiveGame::new(shard.config.seed, id, shard.full_search_rate),
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
                let outcome = game.collect(
                    shard.settings,
                    shard.config,
                    shard.fast_visits,
                    shard.full_search_rate,
                    |leaf| self.leaves.push((i, leaf)),
                );
                if let Some(value) = outcome {
                    self.finished.push((i, value));
                }
            }
            if !self.leaves.is_empty() {
                return Some(
                    self.leaves
                        .iter()
                        .map(|(_, leaf)| (leaf.board_input(), leaf.packed_moves().collect()))
                        .collect(),
                );
            }
        }
    }
}

struct Request {
    worker: usize,
    group: usize,
    inputs: Vec<Input>,
}

type Reply = (usize, Vec<Prediction>);

fn search(
    worker: usize,
    mut groups: Vec<Group>,
    shard: &Shard,
    ready: mpsc::SyncSender<Request>,
    replies: mpsc::Receiver<Reply>,
) -> Vec<(usize, PlayedGame)> {
    // Prime both groups. While one waits for inference, the worker can advance the other.
    let mut live = groups.len();
    let mut initial = 0..live;
    while live > 0 {
        let (group, predictions) = if let Some(i) = initial.next() {
            (i, Vec::new())
        } else if let Ok(reply) = replies.recv() {
            reply
        } else {
            break;
        };
        if let Some(inputs) = groups[group].advance(predictions, shard) {
            if ready
                .send(Request {
                    worker,
                    group,
                    inputs,
                })
                .is_err()
            {
                break;
            }
        } else {
            live -= 1;
        }
    }
    groups
        .into_iter()
        .flat_map(|group| group.completed)
        .collect()
}

fn infer(
    ready: mpsc::Receiver<Request>,
    replies: Vec<mpsc::Sender<Reply>>,
    mut evaluate: impl FnMut(Vec<Input>) -> io::Result<Vec<Prediction>>,
) -> io::Result<()> {
    while let Ok(first) = ready.recv() {
        let mut inputs = first.inputs;
        let mut tickets = vec![(first.worker, first.group, inputs.len())];
        for request in ready.try_iter() {
            tickets.push((request.worker, request.group, request.inputs.len()));
            inputs.extend(request.inputs);
        }
        let count = inputs.len();
        let predictions = evaluate(inputs)?;
        if predictions.len() != count {
            return Err(io::Error::other("expected one prediction per leaf"));
        }
        let mut predictions = predictions.into_iter();
        for (worker, group, n) in tickets {
            replies[worker]
                .send((group, predictions.by_ref().take(n).collect()))
                .map_err(|_| io::Error::other("search worker stopped"))?;
        }
    }
    Ok(())
}

pub(super) fn play_games(
    settings: &Settings,
    config: &Config,
    mut ids: Range<usize>,
    evaluate: impl FnMut(Vec<Input>) -> io::Result<Vec<Prediction>>,
) -> io::Result<Vec<(usize, PlayedGame)>> {
    let slots = config.batch.get().min(ids.len());
    let workers = config.search_threads.get().min(slots);
    let (fast_visits, full_search_rate) = match config.playout_cap {
        PlayoutCap::Disabled {} => (config.visits_per_move, 1.0),
        PlayoutCap::Enabled {
            fast_visits,
            full_search_rate,
        } => (fast_visits, full_search_rate),
    };
    let groups: Vec<_> = (0..workers)
        .map(|worker| {
            let size = slots / workers + usize::from(worker < slots % workers);
            [size.div_ceil(2), size / 2]
                .into_iter()
                .filter(|&n| n > 0)
                .map(|n| Group {
                    active: ids
                        .by_ref()
                        .take(n)
                        .map(|id| ActiveGame::new(config.seed, id, full_search_rate))
                        .collect(),
                    ..Group::default()
                })
                .collect()
        })
        .collect();
    let shard = Shard {
        settings,
        config,
        fast_visits,
        full_search_rate,
        remaining: Mutex::new(ids),
    };
    thread::scope(|scope| {
        let (ready, requests) = mpsc::sync_channel(workers.saturating_mul(2).min(slots));
        let mut senders = Vec::new();
        let mut handles = Vec::new();
        for (i, owned) in groups.into_iter().enumerate() {
            let (tx, rx) = mpsc::channel();
            senders.push(tx);
            let (shard, ready) = (&shard, ready.clone());
            handles.push(scope.spawn(move || search(i, owned, shard, ready, rx)));
        }
        drop(ready);
        // infer owns the channels. An inference error wakes every blocked worker before joining.
        infer(requests, senders, evaluate)?;
        let mut games = Vec::new();
        for worker in handles {
            games.extend(worker.join().expect("self-play worker panicked"));
        }
        games.sort_unstable_by_key(|(id, _)| *id);
        Ok(games)
    })
}

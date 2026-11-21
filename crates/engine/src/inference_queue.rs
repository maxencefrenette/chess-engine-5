use crate::model_evaluator::Input;
use chess_network::inference::Prediction;
use std::{io, num::NonZeroUsize, sync::mpsc, thread};

struct Request {
    worker: usize,
    group: usize,
    model: usize,
    inputs: Vec<Input>,
}

type Reply = (usize, Vec<Prediction>);

fn search<G>(
    worker: usize,
    mut groups: Vec<G>,
    advance: &impl Fn(&mut G, Vec<Prediction>) -> Option<(usize, Vec<Input>)>,
    ready: mpsc::SyncSender<Request>,
    replies: mpsc::Receiver<Reply>,
) -> Vec<G> {
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
        if let Some((model, inputs)) = advance(&mut groups[group], predictions) {
            if ready
                .send(Request {
                    worker,
                    group,
                    model,
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
}

fn infer(
    ready: mpsc::Receiver<Request>,
    replies: Vec<mpsc::Sender<Reply>>,
    mut evaluate: impl FnMut(usize, Vec<Input>) -> io::Result<Option<Vec<Prediction>>>,
) -> io::Result<()> {
    #[derive(Default)]
    struct Batch {
        inputs: Vec<Input>,
        tickets: Vec<(usize, usize, usize)>,
    }

    while let Ok(first) = ready.recv() {
        let mut batches: [Batch; 2] = Default::default();
        for request in std::iter::once(first).chain(ready.try_iter()) {
            let batch = &mut batches[request.model];
            batch
                .tickets
                .push((request.worker, request.group, request.inputs.len()));
            if batch.inputs.is_empty() {
                batch.inputs = request.inputs;
            } else {
                batch.inputs.extend(request.inputs);
            }
        }
        for (model, Batch { inputs, tickets }) in batches.into_iter().enumerate() {
            let count = inputs.len();
            if count == 0 {
                continue;
            }
            let Some(predictions) = evaluate(model, inputs)? else {
                return Ok(());
            };
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
    }
    Ok(())
}

/// Each worker owns two groups of games; at most one request per group is outstanding.
/// Returning None from the evaluator ends a match and wakes the waiting workers.
pub(crate) fn run<G: Send>(
    slots: usize,
    search_threads: NonZeroUsize,
    mut new_group: impl FnMut(usize) -> G,
    advance: impl Fn(&mut G, Vec<Prediction>) -> Option<(usize, Vec<Input>)> + Sync,
    evaluate: impl FnMut(usize, Vec<Input>) -> io::Result<Option<Vec<Prediction>>>,
) -> io::Result<Vec<G>> {
    let workers = search_threads.get().min(slots);
    let groups: Vec<_> = (0..workers)
        .map(|worker| {
            let size = slots / workers + usize::from(worker < slots % workers);
            [size.div_ceil(2), size / 2]
                .into_iter()
                .filter(|&n| n > 0)
                .map(&mut new_group)
                .collect()
        })
        .collect();
    thread::scope(|scope| {
        let (ready, requests) = mpsc::sync_channel(workers.saturating_mul(2).min(slots));
        let mut senders = Vec::new();
        let mut handles = Vec::new();
        for (i, owned) in groups.into_iter().enumerate() {
            let (tx, rx) = mpsc::channel();
            senders.push(tx);
            let (advance, ready) = (&advance, ready.clone());
            handles.push(scope.spawn(move || search(i, owned, advance, ready, rx)));
        }
        drop(ready);
        // infer owns the channels. An inference error wakes every blocked worker before joining.
        infer(requests, senders, evaluate)?;
        let mut groups = Vec::new();
        for worker in handles {
            groups.extend(worker.join().expect("search worker panicked"));
        }
        Ok(groups)
    })
}

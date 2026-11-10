use std::{
    io,
    path::{Path, PathBuf},
};

use chess_config::DeviceConfig;
use chess_network::{
    inference::{InferenceModel, Prediction},
    value::ValueConfig,
};

use crate::{
    chess::{Color, Game},
    mcts::{Leaf, MoveVisits, Search},
};

pub struct ModelEvaluator {
    model: InferenceModel,
    pub(crate) model_dir: PathBuf,
}

impl ModelEvaluator {
    pub fn load(model_dir: &Path, device: DeviceConfig) -> io::Result<Self> {
        Ok(Self {
            model: InferenceModel::load(model_dir, device)?,
            model_dir: model_dir.into(),
        })
    }

    pub(crate) fn value_config(&self) -> ValueConfig {
        self.model.value_config()
    }

    pub(crate) fn evaluate(&self, batch: &[(usize, Leaf)]) -> io::Result<Vec<Prediction>> {
        self.model.evaluate(
            batch.iter().map(|(_, leaf)| {
                (
                    leaf.game.snapshot(&leaf.moves),
                    leaf.game.turn() == Color::Black,
                )
            }),
            batch
                .iter()
                .map(|(_, leaf)| leaf.moves.iter().map(|mv| mv.packed())),
        )
    }

    pub fn search(
        &self,
        game: &Game,
        visits: std::num::NonZeroU32,
        exploration: f32,
    ) -> io::Result<Vec<MoveVisits>> {
        if game.outcome().is_some() {
            return Ok(Vec::new());
        }
        let mut search = Search::new();
        while let Some(leaf) = search.next_leaf(game, visits, exploration) {
            let batch = [(0, leaf)];
            let result = self.evaluate(&batch)?.pop().expect("one prediction");
            let [(_, leaf)] = batch;
            search.evaluate(leaf, result.value, &result.priors);
        }
        Ok(search.visits().collect())
    }
}

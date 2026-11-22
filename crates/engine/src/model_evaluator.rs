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

pub(crate) type Input = ((chess_data::Board, bool), Vec<u16>);

impl Leaf {
    pub(crate) fn input(&self) -> Input {
        (
            (
                self.game.snapshot(&self.moves),
                self.game.turn() == Color::Black,
            ),
            self.moves.iter().map(|mv| mv.packed()).collect(),
        )
    }
}

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

    pub(crate) fn evaluate(&self, batch: Vec<Input>) -> io::Result<Vec<Prediction>> {
        let (boards, moves): (Vec<_>, Vec<_>) = batch.into_iter().unzip();
        self.model.evaluate(
            boards.into_iter(),
            moves.iter().map(|row| row.iter().copied()),
        )
    }

    pub fn search(
        &self,
        game: &Game,
        visits: std::num::NonZeroU32,
        settings: &crate::settings::SearchSettings,
    ) -> io::Result<Vec<MoveVisits>> {
        if game.outcome().is_some() {
            return Ok(Vec::new());
        }
        let mut search = Search::new();
        while let Some(leaf) = search.next_leaf(game, visits, settings) {
            let result = self
                .evaluate(vec![leaf.input()])?
                .pop()
                .expect("one prediction");
            search.evaluate(
                leaf,
                result.value,
                result.draw,
                &result.priors,
                settings.policy_temperature,
            );
        }
        Ok(search.visits().collect())
    }
}

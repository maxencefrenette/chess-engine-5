#[chess_config::toml_config]
pub struct EngineConfig {
    pub search: SearchSettings,
}

#[chess_config::toml_config]
#[derive(Clone, Copy, serde::Serialize)]
pub struct SearchSettings {
    #[range(min = 0.0)]
    pub exploration: f32,
    pub reuse_subtree: bool,
    pub first_play: FirstPlay,
    pub pending_value: PendingValue,
    #[range(min_exclusive = 0.0)]
    pub policy_temperature: f32,
}

#[chess_config::toml_config]
#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum FirstPlay {
    Zero {},
    Parent {
        #[range(min = 0.0, max = 2.0)]
        reduction: f32,
    },
}

#[chess_config::toml_config]
#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum PendingValue {
    Loss {
        #[range(min = 0.0, max = 1.0)]
        amount: f32,
    },
    Mean {},
}

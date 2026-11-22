use std::error::Error;

use chess_config::TomlConfig;

#[chess_config::toml_config]
#[derive(Clone, Copy)]
pub struct Settings {
    #[range(min = 0.0)]
    pub exploration: f32,
    pub reuse_subtree: bool,
}

impl Settings {
    pub fn load() -> Result<Self, Box<dyn Error>> {
        Self::from_repo_file("engine.toml")
    }
}

use std::error::Error;

use chess_config::TomlConfig;

#[chess_config::toml_config]
pub struct Settings {
    pub exploration: f64,
}

impl Settings {
    pub fn load() -> Result<Self, Box<dyn Error>> {
        let settings = Self::from_repo_file("engine.toml")?;
        if !(settings.exploration as f32).is_finite() || settings.exploration < 0.0 {
            return Err("engine setting exploration must be finite and nonnegative".into());
        }
        Ok(settings)
    }
}

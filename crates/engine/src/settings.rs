use std::error::Error;

use chess_config::TomlConfig;

#[chess_config::toml_config]
pub struct Settings {
    pub exploration: f32,
    pub eval_temperature: f64,
    pub eval_sampling_plies: u32,
}

impl Settings {
    pub fn load() -> Result<Self, Box<dyn Error>> {
        let settings: Self = Self::from_file("config/engine.toml")?;
        for (key, value) in [
            ("exploration", f64::from(settings.exploration)),
            ("eval_temperature", settings.eval_temperature),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("engine setting {key} must be finite and nonnegative").into());
            }
        }
        Ok(settings)
    }
}

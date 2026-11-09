use std::{error::Error, iter};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub exploration: f32,
    pub self_play_temperature: f64,
    pub eval_temperature: f64,
    pub eval_sampling_plies: u32,
}

impl Settings {
    pub fn load() -> Result<Self, Box<dyn Error>> {
        let settings: Self = chess_config::load("config/engine.toml", iter::empty())?;
        for (key, value) in [
            ("exploration", f64::from(settings.exploration)),
            ("self_play_temperature", settings.self_play_temperature),
            ("eval_temperature", settings.eval_temperature),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("engine setting {key} must be finite and nonnegative").into());
            }
        }
        Ok(settings)
    }
}

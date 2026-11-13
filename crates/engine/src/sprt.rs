// Scores are the candidate's total points per game pair, in half-points.
mod analytics;
mod likelihood;

#[cfg(test)]
mod tests;

use std::io;

pub(crate) use analytics::Estimate;
use likelihood::Distribution;

const NORMALIZATION: f64 = 800.0 / std::f64::consts::LN_10;

#[chess_config::toml_config]
pub(crate) struct Config {
    pub lower_elo: f64,
    pub upper_elo: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl Config {
    pub fn validate(&self) -> bool {
        self.lower_elo.is_finite()
            && self.upper_elo.is_finite()
            && self.lower_elo < self.upper_elo
            && self.alpha > 0.0
            && self.beta > 0.0
            && self.alpha + self.beta < 1.0
    }

    pub fn boundaries(&self) -> (f64, f64) {
        (
            (self.beta / (1.0 - self.alpha)).ln(),
            ((1.0 - self.beta) / self.alpha).ln(),
        )
    }

    pub fn llr(&self, counts: [u64; 5]) -> io::Result<f64> {
        if counts == [0; 5] {
            return Ok(0.0);
        }
        let distribution = Distribution::new(counts);
        let fit = |elo| distribution.fit_standardized(elo / NORMALIZATION * 2.0_f64.sqrt());
        let lower = fit(self.lower_elo)?;
        let upper = fit(self.upper_elo)?;
        let llr = distribution.count
            * (0..5)
                .map(|i| distribution.probabilities[i] * (upper[i].ln() - lower[i].ln()))
                .sum::<f64>();
        if !llr.is_finite() {
            return Err(io::Error::other("non-finite SPRT likelihood"));
        }
        Ok(llr)
    }

    /// Fishtest's Brownian approximation, accounting for the SPRT boundaries.
    /// Bounds use normalized Elo; estimates and intervals use ordinary Elo.
    pub fn estimate(&self, counts: [u64; 5]) -> Option<Estimate> {
        analytics::estimate(self, counts)
    }
}

/// Fixed-sample normal approximation over game-pair scores.
pub(crate) fn fixed_estimate(counts: [u64; 5]) -> Option<Estimate> {
    let pairs = counts.iter().sum::<u64>() as f64;
    if pairs < 2.0 {
        return None;
    }
    let (mean, variance) = likelihood::moments(counts.map(|n| n as f64 / pairs));
    if variance == 0.0 {
        return None;
    }
    let margin = 1.959963984540054 * (variance / (pairs - 1.0)).sqrt();
    Some(Estimate {
        elo: score_to_elo(mean),
        interval: [mean - margin, mean + margin].map(|score| score_to_elo(score.clamp(0.0, 1.0))),
    })
}

fn score_to_elo(score: f64) -> f64 {
    400.0 * (score / (1.0 - score)).log10()
}

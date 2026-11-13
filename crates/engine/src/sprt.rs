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
    pub lower_nelo: f64,
    pub upper_nelo: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl Config {
    pub fn validate(&self) -> bool {
        self.lower_nelo.is_finite()
            && self.upper_nelo.is_finite()
            && self.lower_nelo < self.upper_nelo
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
        let fit = |nelo| distribution.fit_standardized(nelo / NORMALIZATION * 2.0_f64.sqrt());
        let lower = fit(self.lower_nelo)?;
        let upper = fit(self.upper_nelo)?;
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
    /// Bounds, estimates, and intervals use nElo.
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
        nelo: score_to_nelo(mean, variance),
        interval: [mean - margin, mean + margin]
            .map(|score| score_to_nelo(score.clamp(0.0, 1.0), variance)),
    })
}

fn score_to_nelo(score: f64, pair_variance: f64) -> f64 {
    NORMALIZATION * (score - 0.5) / (2.0 * pair_variance).sqrt()
}

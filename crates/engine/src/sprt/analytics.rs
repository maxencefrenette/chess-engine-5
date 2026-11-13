// Equations and approximations follow Fishtest stats/sprt.py and brownian.py.
use std::f64::consts::PI;

use super::{
    Config, NORMALIZATION,
    likelihood::{Distribution, moments},
    score_to_nelo,
};

pub(crate) struct Estimate {
    pub nelo: f64,
    pub interval: [f64; 2],
}

pub(super) fn estimate(config: &Config, counts: [u64; 5]) -> Option<Estimate> {
    // Do not turn pseudocounts into confidence for a degenerate real sample.
    if counts.iter().sum::<u64>() < 2 || counts.iter().filter(|&&n| n > 0).count() < 2 {
        return None;
    }
    let distribution = Distribution::new(counts);
    let (mean, variance) = moments(distribution.probabilities);
    let sigma_pg = (2.0 * variance).sqrt();
    let to_score = |nelo| 0.5 + nelo / NORMALIZATION * sigma_pg;
    let low = to_score(config.lower_nelo);
    let high = to_score(config.upper_nelo);
    let midpoint = (low + high) / 2.0;
    let slope = (high - low) / variance;
    let drift = (mean - midpoint) * slope;
    let diffusion = (high - low).powi(2) / variance;
    let (a, b) = config.boundaries();
    let mut time = distribution.count;
    let mut llr = time * drift;
    // Fishtest clamps the approximate path to its first boundary crossing.
    if llr < a {
        time = a / drift;
        llr = a;
    } else if llr > b {
        time = b / drift;
        llr = b;
    }
    let outcome = |score| cdf(a, b, (score - midpoint) * slope, diffusion, time, llr);
    let quantile = |p: f64| {
        let mut lo = 0.0;
        let mut hi = 1.0;
        let target = 1.0 - p;
        // Report unavailable rather than inventing finite endpoints at a search limit.
        if outcome(lo)? < target || outcome(hi)? > target {
            return None;
        }
        for _ in 0..64 {
            let mid = (lo + hi) / 2.0;
            if outcome(mid)? > target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(score_to_nelo((lo + hi) / 2.0, variance))
    };
    Some(Estimate {
        nelo: quantile(0.5)?,
        interval: [quantile(0.025)?, quantile(0.975)?],
    })
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x / 2.0_f64.sqrt())
}

// Probability of an outcome at or below this point on a bounded Brownian path.
fn cdf(a: f64, b: f64, drift: f64, variance: f64, time: f64, value: f64) -> Option<f64> {
    let width = b - a;
    let gamma = drift / variance;
    let result = if variance * time / width.powi(2) < 1e-2 || (gamma * width).abs() > 15.0 {
        let denominator = (time * variance).sqrt();
        let offset = drift * time;
        let z = (value - offset) / denominator;
        let za = (-value + offset + 2.0 * a) / denominator;
        let zb = (value - offset - 2.0 * b) / denominator;
        let tail = |z: f64, boundary: f64| {
            if gamma * boundary >= 5.0 {
                -(-z * z / 2.0 + 2.0 * gamma * boundary).exp() / (2.0 * PI).sqrt()
                    * (1.0 / z - 1.0 / z.powi(3))
            } else {
                (2.0 * gamma * boundary).exp() * normal_cdf(z)
            }
        };
        normal_cdf(z) + tail(za, a) - tail(zb, b)
    } else {
        let x = -a;
        let y = value - a;
        let first = (PI / width).powi(2) * variance / 2.0 + drift.powi(2) / variance / 2.0;
        let factor = (-first * time - x * gamma + y * gamma).exp();
        let mut sum = 0.0;
        let mut converged = false;
        for n in 1..=10000 {
            let npi = n as f64 * PI;
            let eigenvalue =
                (npi / width).powi(2) * variance / 2.0 + drift.powi(2) / variance / 2.0;
            let decay = (-(eigenvalue - first) * time).exp();
            let primitive = (2.0 * width * gamma * (npi * y / width).sin()
                - 2.0 * npi * (npi * y / width).cos())
                / (width.powi(2) * gamma.powi(2) + npi.powi(2));
            sum += decay * primitive * (npi * x / width).sin();
            if (factor * decay * primitive).abs() <= 1e-9 {
                converged = true;
                break;
            }
        }
        if !converged {
            return None;
        }
        let eventual = if (gamma * width).abs() < 1e-8 {
            (width - x) / width
        } else {
            (2.0 * gamma * (width - x)).exp_m1() / (2.0 * gamma * width).exp_m1()
        };
        eventual + factor * sum
    };
    (result.is_finite() && (-0.001..=1.001).contains(&result)).then(|| result.clamp(0.0, 1.0))
}

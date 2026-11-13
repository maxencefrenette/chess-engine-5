use std::io;

pub(super) struct Distribution {
    pub count: f64,
    pub probabilities: [f64; 5],
}

pub(super) fn moments(probabilities: [f64; 5]) -> (f64, f64) {
    let mean: f64 = (0..5).map(|i| i as f64 / 4.0 * probabilities[i]).sum();
    let variance = (0..5)
        .map(|i| probabilities[i] * (i as f64 / 4.0 - mean).powi(2))
        .sum();
    (mean, variance)
}

impl Distribution {
    pub fn new(counts: [u64; 5]) -> Self {
        // Match Fishtest's treatment of empty bins.
        let weights = counts.map(|n| if n == 0 { 0.001 } else { n as f64 });
        let count = weights.iter().sum();
        Self {
            count,
            probabilities: weights.map(|n| n / count),
        }
    }

    pub fn fit_standardized(&self, target: f64) -> io::Result<[f64; 5]> {
        let mut probabilities = [0.2; 5];
        // Fishtest's iterative constrained maximum-likelihood fit.
        for _ in 0..10 {
            let (mean, variance) = moments(probabilities);
            let sigma = variance.sqrt();
            let fitted = self.fit(std::array::from_fn(|i| {
                let value = i as f64 / 4.0;
                value - 0.5 - target * sigma * (1.0 + ((mean - value) / sigma).powi(2)) / 2.0
            }))?;
            let change = (0..5)
                .map(|i| (fitted[i] - probabilities[i]).abs())
                .fold(0.0, f64::max);
            probabilities = fitted;
            if change < 1e-9 {
                break;
            }
        }
        let (mean, variance) = moments(probabilities);
        if ((mean - 0.5) / variance.sqrt() - target).abs() >= 1e-5 {
            return Err(io::Error::other("normalized SPRT fit did not converge"));
        }
        Ok(probabilities)
    }

    fn fit(&self, distances: [f64; 5]) -> io::Result<[f64; 5]> {
        let min = distances.into_iter().fold(f64::INFINITY, f64::min);
        let max = distances.into_iter().fold(f64::NEG_INFINITY, f64::max);
        if min >= 0.0 || max <= 0.0 {
            return Err(io::Error::other(
                "SPRT hypothesis lies outside score support",
            ));
        }
        let mut low = -1.0 / max;
        let mut high = -1.0 / min;
        for _ in 0..80 {
            let mid = (low + high) / 2.0;
            let residual: f64 = (0..5)
                .map(|i| self.probabilities[i] * distances[i] / (1.0 + mid * distances[i]))
                .sum();
            if residual > 0.0 {
                low = mid;
            } else {
                high = mid;
            }
        }
        let lambda = (low + high) / 2.0;
        let fitted = std::array::from_fn(|i| self.probabilities[i] / (1.0 + lambda * distances[i]));
        if fitted.iter().any(|p| !p.is_finite() || *p <= 0.0) {
            return Err(io::Error::other("invalid SPRT fitted probability"));
        }
        Ok(fitted)
    }
}

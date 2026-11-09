// Scores are the candidate's total points per color-swapped pair, in half-points.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
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
            && score(self.lower_elo) > 0.0
            && score(self.upper_elo) < 1.0
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

    pub fn llr(&self, counts: [u64; 5]) -> f64 {
        log_likelihood(counts, self.upper_elo) - log_likelihood(counts, self.lower_elo)
    }
}

fn score(elo: f64) -> f64 {
    1.0 / (1.0 + 10_f64.powf(-elo / 400.0))
}

fn log_likelihood(counts: [u64; 5], elo: f64) -> f64 {
    let score = score(elo);
    let distances = std::array::from_fn::<_, 5, _>(|i| i as f64 / 4.0 - score);
    // Regularize unobserved bins so the constrained solution has full support.
    let weights = counts.map(|n| if n == 0 { 0.001 } else { n as f64 });
    let mut low = -1.0 / distances[4];
    let mut high = -1.0 / distances[0];
    // Profile the unknown pair distribution subject to the hypothesized mean.
    // Its fitted probabilities are proportional to n_i / (1 + lambda * d_i).
    for _ in 0..64 {
        let mid = (low + high) / 2.0;
        let residual: f64 = weights
            .iter()
            .zip(distances)
            .map(|(weight, d)| weight * d / (1.0 + mid * d))
            .sum();
        if residual > 0.0 {
            low = mid;
        } else {
            high = mid;
        }
    }
    let lambda = (low + high) / 2.0;
    // The empirical-frequency terms cancel when subtracting the two fits.
    -weights
        .iter()
        .zip(distances)
        .map(|(weight, d)| weight * (lambda * d).ln_1p())
        .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logistic_pair_likelihood_matches_closed_form_for_balanced_decisive_pairs() {
        // With only the two extreme bins occupied, the fitted probabilities
        // approach those of a Bernoulli trial. Regularization is negligible here.
        let config = Config {
            lower_elo: 0.,
            upper_elo: 20.,
            alpha: 0.05,
            beta: 0.05,
        };
        let score = score(config.upper_elo);
        let (lower, upper) = config.boundaries();
        let expected = 1000.0 * (4.0 * score * (1.0 - score)).ln();
        assert!((config.llr([1000, 0, 0, 0, 1000]) - expected).abs() < 0.001);
        assert!(config.llr([0, 0, 0, 0, 200]) > upper);
        assert!(config.llr([200, 0, 0, 0, 0]) < lower);
    }
}

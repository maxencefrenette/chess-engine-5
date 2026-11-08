// Scores are the candidate's total points per color-swapped pair, in half-points.
pub(crate) const BOUNDARY: f64 = 2.944_438_979_166_440_3;

fn log_likelihood(counts: [u64; 5], elo: f64) -> f64 {
    let score = 1.0 / (1.0 + 10_f64.powf(-elo / 400.0));
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

pub(crate) fn llr(counts: [u64; 5]) -> f64 {
    log_likelihood(counts, 10.0) - log_likelihood(counts, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logistic_pair_likelihood_matches_closed_form_for_balanced_decisive_pairs() {
        // With only the two extreme bins occupied, the fitted probabilities
        // approach those of a Bernoulli trial. Regularization is negligible here.
        let score = 1.0 / (1.0 + 10_f64.powf(-10.0 / 400.0));
        let expected = 1000.0 * (4.0 * score * (1.0 - score)).ln();
        assert!((llr([1000, 0, 0, 0, 1000]) - expected).abs() < 0.001);
        assert!(llr([0, 0, 0, 0, 200]) > BOUNDARY);
        assert!(llr([200, 0, 0, 0, 0]) < -BOUNDARY);
    }
}

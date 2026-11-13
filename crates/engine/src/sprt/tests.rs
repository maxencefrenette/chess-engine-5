use super::*;

fn check_fishtest_reference(
    bounds: [f64; 2],
    errors: [f64; 2],
    counts: [u64; 5],
    expected_llr: f64,
    expected_estimate: Option<[f64; 3]>,
) {
    let config = Config {
        lower_nelo: bounds[0],
        upper_nelo: bounds[1],
        alpha: errors[0],
        beta: errors[1],
    };
    let llr = config.llr(counts).unwrap();
    assert!(
        (llr - expected_llr).abs() < 1e-5,
        "LLR: {llr} != {expected_llr}"
    );
    let (lower, upper) = config.boundaries();
    let verdict = |value| {
        if value <= lower {
            -1
        } else if value >= upper {
            1
        } else {
            0
        }
    };
    assert_eq!(verdict(llr), verdict(expected_llr));
    match (config.estimate(counts), expected_estimate) {
        (Some(actual), Some(expected)) => {
            for (actual, expected) in [actual.nelo, actual.interval[0], actual.interval[1]]
                .into_iter()
                .zip(expected)
            {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "estimate: {actual} != {expected}"
                );
            }
        }
        (None, None) => {}
        _ => panic!("missing estimate"),
    }
}

macro_rules! fishtest_cases {
    ($($name:ident: ($bounds:expr, $errors:expr, $counts:expr, $llr:expr, $estimate:expr);)*) => {
        $(#[test]
        fn $name() {
            check_fishtest_reference($bounds, $errors, $counts, $llr, $estimate);
        })*
    };
}

// Reference: https://github.com/official-stockfish/fishtest/tree/f7af98f81f0c740a4de3d9a3721854709dcdc2bb/server/fishtest/stats
// Generated with LLRcalc.LLR_normalized and sprt.sprt.analytics (Python 3.12.3, SciPy 1.18.1).
// Estimates and 95% intervals converted with Fishtest sprt.lelo_to_elo; all values are nElo.
fishtest_cases! {
    gainer_winning: ([0.0, 20.0], [0.05, 0.05], [135, 440, 682, 529, 214], 11.906324907396973, Some([24.773518501829297, 1.6945603359682657, 47.17242461985012]));
    gainer_losing: ([0.0, 20.0], [0.05, 0.05], [214, 529, 682, 440, 135], -25.088466205134267, Some([-24.732108026743333, -56.78804523997083, 7.671434948186926]));
    gainer_balanced: ([0.0, 20.0], [0.05, 0.05], [100, 200, 400, 200, 100], -3.310935170695074, Some([3.2452944311975167, -13.733495329621569, 21.308449383222683]));
    gainer_small_sample: ([0.0, 20.0], [0.05, 0.05], [1, 2, 5, 2, 1], -0.036435285190315476, Some([1.0445644497232271e-13, -145.18142302112634, 145.18142302112668]));
    gainer_draw_heavy: ([0.0, 20.0], [0.05, 0.05], [0, 20, 960, 20, 0], -3.433286941365714, Some([3.2452944311972742, -13.733495329621784, 21.308449383223042]));
    gainer_all_even_pairs: ([0.0, 20.0], [0.05, 0.05], [0, 0, 100, 0, 0], -0.6538942632756706, None);
    gainer_all_losses: ([0.0, 20.0], [0.05, 0.05], [200, 0, 0, 0, 0], -16.92440653822364, None);
    gainer_all_wins: ([0.0, 20.0], [0.05, 0.05], [0, 0, 0, 0, 200], 15.603284254379576, None);
    gainer_sparse: ([0.0, 20.0], [0.05, 0.05], [0, 0, 0, 2, 1], 0.18952047854347304, Some([684.6611251894115, 406.79876176954804, 962.5234886092749]));
    non_regression_winning: ([-20.0, 0.0], [0.05, 0.05], [135, 440, 682, 529, 214], 25.088466205134267, Some([24.732108026743333, -7.671434948186876, 56.78804523997088]));
    non_regression_losing: ([-20.0, 0.0], [0.05, 0.05], [214, 529, 682, 440, 135], -11.906324907396876, Some([-24.773518501829248, -47.17242461985007, -1.6945603359683161]));
    non_regression_balanced: ([-20.0, 0.0], [0.05, 0.05], [100, 200, 400, 200, 100], 3.3109351706950285, Some([-3.2452944311975167, -21.308449383222786, 13.733495329621467]));
    non_regression_small_sample: ([-20.0, 0.0], [0.05, 0.05], [1, 2, 5, 2, 1], 0.036435285190315476, Some([0.0, -145.18142302112668, 145.18142302112634]));
    non_regression_draw_heavy: ([-20.0, 0.0], [0.05, 0.05], [0, 20, 960, 20, 0], 3.433286941365714, Some([-3.2452944311978196, -21.308449383223316, 13.733495329621238]));
    non_regression_all_even_pairs: ([-20.0, 0.0], [0.05, 0.05], [0, 0, 100, 0, 0], 0.6538942632756706, None);
    non_regression_all_losses: ([-20.0, 0.0], [0.05, 0.05], [200, 0, 0, 0, 0], -15.603284254379563, None);
    non_regression_all_wins: ([-20.0, 0.0], [0.05, 0.05], [0, 0, 0, 0, 200], 16.924406538223618, None);
    non_regression_sparse: ([-20.0, 0.0], [0.05, 0.05], [0, 0, 0, 2, 1], 0.2089410607506782, Some([684.6611251894115, 406.7987617695471, 962.5234886092749]));
    asymmetric_errors_winning: ([0.0, 5.0], [0.01, 0.1], [135, 440, 682, 529, 214], 4.214161726817676, Some([28.05081821030327, 17.276401965664828, 38.82288688314234]));
    asymmetric_errors_losing: ([0.0, 5.0], [0.01, 0.1], [214, 529, 682, 440, 135], -5.038874986206401, Some([-26.999810195516663, -43.125436192968124, -10.82813411726872]));
    asymmetric_errors_balanced: ([0.0, 5.0], [0.01, 0.1], [100, 200, 400, 200, 100], -0.20709467029953474, Some([1.0098964686151437e-10, -15.226756105369464, 15.226756105586134]));
    asymmetric_errors_small_sample: ([0.0, 5.0], [0.01, 0.1], [1, 2, 5, 2, 1], -0.002278100326254867, Some([0.0, -145.18142302112668, 145.18142302112668]));
    asymmetric_errors_draw_heavy: ([0.0, 5.0], [0.01, 0.1], [0, 20, 960, 20, 0], -0.20757763659263817, Some([1.0090861921200204e-10, -15.226740878636006, 15.226740878852825]));
    asymmetric_errors_all_even_pairs: ([0.0, 5.0], [0.01, 0.1], [0, 0, 100, 0, 0], -0.03753988283646515, None);
    asymmetric_errors_all_losses: ([0.0, 5.0], [0.01, 0.1], [200, 0, 0, 0, 0], -4.111576210601552, None);
    asymmetric_errors_all_wins: ([0.0, 5.0], [0.01, 0.1], [0, 0, 0, 0, 200], 4.028749910752937, None);
    asymmetric_errors_sparse: ([0.0, 5.0], [0.01, 0.1], [0, 0, 0, 2, 1], 0.04922667869907793, Some([684.6611251894115, 406.79876176954804, 962.5234886092749]));
}

#[test]
fn validates_config_and_handles_no_results() {
    let mut config = Config {
        lower_nelo: 0.,
        upper_nelo: 20.,
        alpha: 0.05,
        beta: 0.05,
    };
    assert!(config.validate());
    assert_eq!(config.llr([0; 5]).unwrap(), 0.0);
    assert!(config.estimate([0; 5]).is_none());
    assert!(config.estimate([0, 0, 1, 0, 0]).is_none());
    config.upper_nelo = 0.;
    assert!(!config.validate());
    config.upper_nelo = f64::NAN;
    assert!(!config.validate());
    config.upper_nelo = 20.;
    config.alpha = 0.;
    assert!(!config.validate());
}

#[test]
fn fixed_nelo_accounts_for_pair_variance() {
    // Same 55% score, but more concentrated pair scores imply greater nElo.
    let spread = fixed_estimate([45, 0, 0, 0, 55]).unwrap();
    let concentrated = fixed_estimate([0, 40, 0, 60, 0]).unwrap();
    assert!((spread.nelo - 24.69117190694986).abs() < 1e-9);
    assert!((concentrated.nelo - 50.1480072068823).abs() < 1e-9);
    let reversed = fixed_estimate([55, 0, 0, 0, 45]).unwrap();
    assert!((spread.nelo + reversed.nelo).abs() < 1e-9);
    assert!((spread.interval[0] + reversed.interval[1]).abs() < 1e-9);
    assert!((spread.interval[1] + reversed.interval[0]).abs() < 1e-9);
}

use super::*;

fn check_fishtest_reference(
    bounds: [f64; 2],
    errors: [f64; 2],
    counts: [u64; 5],
    expected_llr: f64,
    expected_estimate: Option<[f64; 3]>,
) {
    let config = Config {
        lower_elo: bounds[0],
        upper_elo: bounds[1],
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
            for (actual, expected) in [actual.elo, actual.interval[0], actual.interval[1]]
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
// Bounds are normalized Elo; estimates and 95% intervals are ordinary Elo.
fishtest_cases! {
    gainer_winning: ([0.0, 20.0], [0.05, 0.05], [135, 440, 682, 529, 214], 11.906324907396973, Some([18.94455157022488, 1.294570098261024, 36.16758565010445]));
    gainer_losing: ([0.0, 20.0], [0.05, 0.05], [214, 529, 682, 440, 135], -25.088466205134267, Some([-18.912821928261835, -43.61099517907812, 5.8611701278185375]));
    gainer_balanced: ([0.0, 20.0], [0.05, 0.05], [100, 200, 400, 200, 100], -3.310935170695074, Some([2.51383812361301, -10.641245911026537, 16.5178876424074]));
    gainer_small_sample: ([0.0, 20.0], [0.05, 0.05], [1, 2, 5, 2, 1], -0.036435285190315476, Some([5.8706521614886e-14, -110.83650420172764, 110.83650420172788]));
    gainer_draw_heavy: ([0.0, 20.0], [0.05, 0.05], [0, 20, 960, 20, 0], -3.433286941365714, Some([0.45899964098383433, -1.942422041673596, 3.0138437218347134]));
    gainer_all_even_pairs: ([0.0, 20.0], [0.05, 0.05], [0, 0, 100, 0, 0], -0.6538942632756706, None);
    gainer_all_losses: ([0.0, 20.0], [0.05, 0.05], [200, 0, 0, 0, 0], -16.92440653822364, None);
    gainer_all_wins: ([0.0, 20.0], [0.05, 0.05], [0, 0, 0, 0, 200], 15.603284254379576, None);
    gainer_sparse: ([0.0, 20.0], [0.05, 0.05], [0, 0, 0, 2, 1], 0.18952047854347304, Some([278.86013282279106, 145.2982700793804, 591.1375213726313]));
    non_regression_winning: ([-20.0, 0.0], [0.05, 0.05], [135, 440, 682, 529, 214], 25.088466205134267, Some([18.912821928261845, -5.86117012781855, 43.61099517907819]));
    non_regression_losing: ([-20.0, 0.0], [0.05, 0.05], [214, 529, 682, 440, 135], -11.906324907396876, Some([-18.944551570224885, -36.167585650104385, -1.2945700982610644]));
    non_regression_balanced: ([-20.0, 0.0], [0.05, 0.05], [100, 200, 400, 200, 100], 3.3109351706950285, Some([-2.5138381236129526, -16.51788764240738, 10.641245911026495]));
    non_regression_small_sample: ([-20.0, 0.0], [0.05, 0.05], [1, 2, 5, 2, 1], 0.036435285190315476, Some([-2.039165159021779e-14, -110.83650420172785, 110.83650420172765]));
    non_regression_draw_heavy: ([-20.0, 0.0], [0.05, 0.05], [0, 20, 960, 20, 0], 3.433286941365714, Some([-0.4589996409838295, -3.013843721834726, 1.942422041673531]));
    non_regression_all_even_pairs: ([-20.0, 0.0], [0.05, 0.05], [0, 0, 100, 0, 0], 0.6538942632756706, None);
    non_regression_all_losses: ([-20.0, 0.0], [0.05, 0.05], [200, 0, 0, 0, 0], -15.603284254379563, None);
    non_regression_all_wins: ([-20.0, 0.0], [0.05, 0.05], [0, 0, 0, 0, 200], 16.924406538223618, None);
    non_regression_sparse: ([-20.0, 0.0], [0.05, 0.05], [0, 0, 0, 2, 1], 0.2089410607506782, Some([278.8601328227911, 145.29827007938005, 591.1375213726315]));
    asymmetric_errors_winning: ([0.0, 5.0], [0.01, 0.1], [135, 440, 682, 529, 214], 4.214161726817676, Some([21.456738006534028, 13.204709865145691, 29.731212874564225]));
    asymmetric_errors_losing: ([0.0, 5.0], [0.01, 0.1], [214, 529, 682, 440, 135], -5.038874986206401, Some([-20.650865136150443, -33.04508221549538, -8.273746983471847]));
    asymmetric_errors_balanced: ([0.0, 5.0], [0.01, 0.1], [100, 200, 400, 200, 100], -0.20709467029953474, Some([7.818649454374518e-11, -11.799128542994463, 11.799128543162526]));
    asymmetric_errors_small_sample: ([0.0, 5.0], [0.01, 0.1], [1, 2, 5, 2, 1], -0.002278100326254867, Some([-1.0280405396554247e-14, -110.83650420172785, 110.83650420172792]));
    asymmetric_errors_draw_heavy: ([0.0, 5.0], [0.01, 0.1], [0, 20, 960, 20, 0], -0.20757763659263817, Some([1.426445313040721e-11, -2.153627102988234, 2.1536271030189043]));
    asymmetric_errors_all_even_pairs: ([0.0, 5.0], [0.01, 0.1], [0, 0, 100, 0, 0], -0.03753988283646515, None);
    asymmetric_errors_all_losses: ([0.0, 5.0], [0.01, 0.1], [200, 0, 0, 0, 0], -4.111576210601552, None);
    asymmetric_errors_all_wins: ([0.0, 5.0], [0.01, 0.1], [0, 0, 0, 0, 200], 4.028749910752937, None);
    asymmetric_errors_sparse: ([0.0, 5.0], [0.01, 0.1], [0, 0, 0, 2, 1], 0.04922667869907793, Some([278.8601328227911, 145.29827007938036, 591.1375213726309]));
}

#[test]
fn validates_config_and_handles_no_results() {
    let mut config = Config {
        lower_elo: 0.,
        upper_elo: 20.,
        alpha: 0.05,
        beta: 0.05,
    };
    assert!(config.validate());
    assert_eq!(config.llr([0; 5]).unwrap(), 0.0);
    assert!(config.estimate([0; 5]).is_none());
    assert!(config.estimate([0, 0, 1, 0, 0]).is_none());
    config.upper_elo = 0.;
    assert!(!config.validate());
    config.upper_elo = f64::NAN;
    assert!(!config.validate());
    config.upper_elo = 20.;
    config.alpha = 0.;
    assert!(!config.validate());
}

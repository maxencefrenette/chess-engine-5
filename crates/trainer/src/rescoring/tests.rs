use super::*;
use crate::data::Result;
use std::path::PathBuf;

fn row(pieces: &[(usize, u8)], black: bool) -> Row {
    let mut board = [0; 32];
    for &(square, piece) in pieces {
        board[square / 2] |= piece << (4 * (square % 2));
    }
    Row {
        game_id: 0,
        position: chess_data::Position {
            ply: u16::from(black),
            snapshot: Board {
                board,
                castling: 0,
                ep_square: 64,
                halfmove_clock: 0,
                repetitions: 1,
            },
            policy: vec![(123, 800)],
            played_move: 123,
        },
        value: 0,
        generation_ns_per_position: Some(42),
    }
}

fn tables() -> Result<Rescorer> {
    Ok(RescoringConfig::Chesstb {
        directory: PathBuf::from("/data/chess/tablebases/chesstb-5"),
        policy_mix: 0.0,
        value_mix: 1.0,
    }
    .open()?)
}

#[test]
fn folds_cursed_and_blessed_classes_into_draws() {
    assert_eq!(
        [
            Wdl::Loss,
            Wdl::BlessedLoss,
            Wdl::Draw,
            Wdl::CursedWin,
            Wdl::Win
        ]
        .map(wdl_target),
        [-1, 0, 0, 0, 1]
    );
}

#[test]
fn serializes_board_orientation_castling_and_en_passant() -> Result<()> {
    let mut row = row(&[(0, 4), (4, 6), (63, 12), (36, 1), (35, 7)], true);
    row.position.snapshot.castling = 2;
    row.position.snapshot.ep_square = 19;
    row.position.snapshot.halfmove_clock = 42;
    assert_eq!(
        fen(&row.position.snapshot, true)?.to_str().unwrap(),
        "7k/8/8/3pP3/8/8/8/R3K3 b Q d3 42 1"
    );
    Ok(())
}

#[test]
fn disabled_rescoring_keeps_data_untouched() -> Result<()> {
    let mut rescorer = RescoringConfig::Disabled.open()?;
    let mut row = row(&[(0, 6), (9, 5), (63, 12)], false);
    rescorer.rescore(&mut row)?;
    assert_eq!(row.value, 0);
    assert_eq!(rescorer.stats.probed, 0);
    Ok(())
}

#[test]
#[ignore = "requires downloaded ChessTB tables on the HDD"]
fn downloaded_tables_respect_side_clock_and_castling() -> Result<()> {
    let mut rescorer = tables()?;
    // KQK, with colors and side to move varied independently.
    for (pieces, black, expected) in [
        ([(0, 6), (9, 5), (47, 12)], false, 1),
        ([(0, 6), (9, 5), (47, 12)], true, -1),
        ([(0, 12), (9, 11), (47, 6)], true, 1),
        ([(0, 12), (9, 11), (47, 6)], false, -1),
    ] {
        let mut row = row(&pieces, black);
        rescorer.rescore(&mut row)?;
        assert_eq!(row.value, expected);
        assert_eq!(row.position.policy, [(123, 800)]);
        assert_eq!(row.generation_ns_per_position, Some(42));
        row.position.snapshot.halfmove_clock = 99;
        rescorer.rescore(&mut row)?;
        assert_eq!(row.value, 0, "not enough time to mate or reset the clock");
    }
    let mut kings = row(&[(0, 6), (63, 12)], false);
    kings.value = 1;
    rescorer.rescore(&mut kings)?;
    assert_eq!(kings.value, 0);
    let mut five = row(&[(0, 6), (9, 5), (17, 5), (25, 5), (47, 12)], false);
    rescorer.rescore(&mut five)?;
    assert_eq!(five.value, 1);
    let mut six = row(
        &[(0, 6), (9, 5), (17, 5), (25, 5), (33, 5), (47, 12)],
        false,
    );
    let probed = rescorer.stats.probed;
    rescorer.rescore(&mut six)?;
    assert_eq!(six.value, 0);
    assert_eq!(rescorer.stats.probed, probed);
    // ChessTB's published castling example: O-O-O+ wins a bishop.
    let mut castle = row(&[(0, 4), (4, 6), (6, 9), (43, 12)], false);
    rescorer.rescore(&mut castle)?;
    assert_eq!(castle.value, 0);
    castle.position.snapshot.castling = 2;
    rescorer.rescore(&mut castle)?;
    assert_eq!(castle.value, 1);
    Ok(())
}

#[test]
fn policy_mix_is_a_direct_fraction_and_preserves_old_boosts() {
    let original = [(1, 60.), (2, 30.), (3, 10.)];
    let mut policy = original;
    blend_policy(&mut policy, &[3], 0.5);
    assert_eq!(policy, [(1, 30.), (2, 15.), (3, 55.)]);
    let mut policy = original;
    blend_policy(&mut policy, &[2, 3], 1.);
    assert_eq!(policy, [(1, 0.), (2, 50.), (3, 50.)]);
    for boost in [0., 0.25, 0.5, 1.] {
        let mut policy = original;
        blend_policy(&mut policy, &[2, 3], boost / (1. + boost));
        for ((m, p), (_, old)) in policy.into_iter().zip(original) {
            let expected = (old / 100. + if m == 1 { 0. } else { boost / 2. }) / (1. + boost);
            assert!((p / 100. - expected).abs() < 1e-7);
        }
        assert!((policy.iter().map(|(_, p)| p).sum::<f32>() - 100.).abs() < 1e-5);
    }
}

#[test]
fn repetition_guard_uses_full_game_before_subsampling() {
    let mut first = row(&[(0, 6), (9, 5), (63, 12)], false);
    first.position.ply = 10;
    first.position.snapshot.repetitions = 2;
    let history = History::new(&[first]);
    let mut later = row(&[(0, 6), (9, 5), (63, 12)], false);
    later.position.ply = 14;
    later.position.snapshot.halfmove_clock = 8;
    assert!(history.has_repeated(&later));
    later.position.snapshot.halfmove_clock = 2;
    assert!(!history.has_repeated(&later));
    later.game_id = 1;
    later.position.snapshot.halfmove_clock = 8;
    assert!(!history.has_repeated(&later));
}

#[test]
#[ignore = "requires downloaded ChessTB tables on the HDD"]
fn dtz_boost_prefers_mate_and_does_not_boost_losses() -> Result<()> {
    let mut rescorer = RescoringConfig::Chesstb {
        directory: "/data/chess/tablebases/chesstb-5".into(),
        policy_mix: 0.5,
        value_mix: 1.0,
    }
    .open()?;
    let mut winning = row(&[(45, 6), (46, 5), (63, 12)], false);
    // Superset permits inspection of native selection without duplicating move generation.
    winning.position.policy = (0..4096).map(|m| (m, 1)).collect();
    let policy = rescorer.policy(&winning)?;
    let mate = 46 | (54 << 6); // Qg7#
    assert!(policy.iter().find(|(m, _)| *m == mate).unwrap().1 > 1.);
    assert_eq!(policy.iter().map(|(_, p)| p).sum::<f32>(), 4096.);
    winning.position.snapshot.halfmove_clock = 99;
    assert!(
        rescorer
            .policy(&winning)?
            .iter()
            .find(|(m, _)| *m == mate)
            .unwrap()
            .1
            > 1.
    );
    let mut losing = row(&[(0, 6), (9, 5), (47, 12)], true);
    losing.position.policy = (0..4096).map(|m| (m, 1)).collect();
    assert!(rescorer.policy(&losing)?.iter().all(|(_, p)| *p == 1.));
    rescorer.rescore_with_history(&mut losing, true)?;
    assert_eq!(losing.value, 0);
    Ok(())
}

#[test]
fn value_mixture_preserves_endpoints_and_probability_mass() {
    for original in -1..=1 {
        for rescored in -1..=1 {
            assert_eq!(blend_value(original, rescored, 0.), game_value(original));
            assert_eq!(blend_value(original, rescored, 1.), game_value(rescored));
            for mix in [0.1, 0.25, 0.5] {
                let target = blend_value(original, rescored, mix);
                assert!((target.iter().sum::<f32>() - 1.).abs() < 1e-6);
                assert!(target.iter().all(|p| (0.0..=1.0).contains(p)));
            }
        }
    }
    assert_eq!(blend_value(0, 1, 0.25), [0., 0.75, 0.25]);
    assert_eq!(blend_value(-1, 0, 0.5), [0.5, 0.5, 0.]);
}

#[test]
#[ignore = "requires downloaded ChessTB tables on the HDD"]
fn soft_rescoring_respects_repetition_guard() -> Result<()> {
    let mut rescorer = RescoringConfig::Chesstb {
        directory: "/data/chess/tablebases/chesstb-5".into(),
        policy_mix: 0.0,
        value_mix: 0.25,
    }
    .open()?;
    let mut winning = row(&[(0, 6), (9, 5), (47, 12)], false);
    assert_eq!(rescorer.value(&mut winning, false)?, [0., 0.75, 0.25]);
    let mut repeated = row(&[(0, 6), (9, 5), (47, 12)], false);
    assert_eq!(rescorer.value(&mut repeated, true)?, [0., 1., 0.]);
    Ok(())
}

#[test]
fn dtz_selection_preserves_ties_and_promotion_encoding() {
    let moves = [
        RootMove {
            dtz: 2,
            from: 8,
            to: 16,
            promotion: 0,
        },
        RootMove {
            dtz: 1,
            from: 48,
            to: 56,
            promotion: 4,
        },
        RootMove {
            dtz: 1,
            from: 49,
            to: 57,
            promotion: 1,
        },
        RootMove {
            dtz: 0,
            from: 9,
            to: 17,
            promotion: 0,
        },
        RootMove {
            dtz: -1,
            from: 10,
            to: 18,
            promotion: 0,
        },
    ];
    assert_eq!(
        boost_moves(&moves, 99),
        [48 | (56 << 6) | (4 << 12), 49 | (57 << 6) | (1 << 12)]
    );
    assert!(boost_moves(&moves[3..], 0).is_empty());
    assert!(
        boost_moves(
            &[RootMove {
                dtz: 1000,
                ..moves[0]
            }],
            0
        )
        .is_empty()
    );
}

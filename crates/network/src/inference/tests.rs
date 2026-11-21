use super::*;
use chess_data::Board;

fn board(i: usize) -> (Board, bool) {
    (
        Board {
            board: std::array::from_fn(|s| (((s + i) % 13) | (((s * 3 + i) % 13) << 4)) as u8),
            castling: (i % 16) as u8,
            ep_square: (i % 65) as u8,
            halfmove_clock: (i % 256) as u8,
            repetitions: (i % 4) as u8,
        },
        i % 2 == 1,
    )
}

#[test]
#[ignore = "requires CUDA and the baseline checkpoint"]
fn cuda_replays_match_eager_with_changing_inputs() -> Result<(), Box<dyn std::error::Error>> {
    let config: toml::Value = toml::from_str(&std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../config/eval-model.toml"
    ))?)?;
    let mut model = InferenceModel::load(
        Path::new(config["baseline_dir"].as_str().unwrap()),
        DeviceConfig::Cuda,
    )?;
    for (iteration, (size, width)) in [
        (512, 19usize),
        (512, 19),
        (17, 9),
        (128, 1),
        (1, 2),
        (2, 4),
        (4, 7),
        (8, 19),
        (16, 65),
        (257, 9),
        (33, 1),
        (512, 2),
        (17, 65),
    ]
    .into_iter()
    .enumerate()
    {
        model.net.config.perspective = if iteration % 2 == 0 {
            crate::perspective::Perspective::SideToMove
        } else {
            crate::perspective::Perspective::Absolute
        };
        let moves: Vec<Vec<u16>> = (0..size)
            .map(|i| {
                (0..width.saturating_sub(i % 3).max(1))
                    .map(|j| ((j % 64) | (((j + 17) % 64) << 6) | ((j % 5) << 12)) as u16)
                    .collect()
            })
            .collect();
        let actual = model.evaluate(
            (0..size).map(|i| board(i + iteration)),
            moves.iter().map(|r| r.iter().copied()),
        )?;
        let mut features = Vec::new();
        let mut network_moves = moves.clone();
        for (i, row) in network_moves.iter_mut().enumerate() {
            let (mut b, mut black) = board(i + iteration);
            let mask = model.net.config.perspective.move_mask(black);
            if mask != 0 {
                b = crate::perspective::flip_board(&b);
                black = false;
                for mv in row {
                    *mv ^= mask;
                }
            }
            encode_position(&mut features, &b, black).unwrap();
        }
        let unpadded = PolicyInputs::new(
            features,
            network_moves.iter().map(|r| r.iter().copied()),
            &model.device,
        )?;
        let (p, v) = unpadded.logits(&model.net)?;
        let unpadded = predictions(
            moves.iter().map(|r| r.iter().copied()),
            p.to_dtype(DType::F32)?.to_vec2()?,
            v.to_dtype(DType::F32)?.to_vec2()?,
        )?;
        assert_eq!(actual.len(), size);
        for (actual, expected) in actual.iter().zip(unpadded) {
            assert_eq!(actual.value, expected.value);
            assert_eq!(actual.draw, expected.draw);
            assert_eq!(actual.priors, expected.priors);
        }
    }
    let invalid = Board {
        board: [255; 32],
        castling: 0,
        ep_square: 64,
        halfmove_clock: 0,
        repetitions: 0,
    };
    assert!(
        model
            .evaluate(
                std::iter::once((invalid, false)),
                std::iter::once(std::iter::once(0u16))
            )
            .is_err()
    );
    assert!(
        model
            .evaluate(
                std::iter::empty(),
                std::iter::empty::<std::iter::Empty<u16>>()
            )?
            .is_empty()
    );
    Ok(())
}

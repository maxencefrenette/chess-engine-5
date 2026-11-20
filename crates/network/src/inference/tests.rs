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
    let model = InferenceModel::load(
        Path::new(config["baseline_dir"].as_str().unwrap()),
        DeviceConfig::Cuda,
    )?;
    for (iteration, size) in [512usize, 512, 17, 128, 1, 2, 4, 8, 16, 257, 33, 512, 17]
        .into_iter()
        .enumerate()
    {
        let width = [19usize, 65, 9][iteration % 3];
        let moves: Vec<Vec<u16>> = (0..size)
            .map(|i| {
                (0..width - i % 3)
                    .map(|j| ((j % 64) | (((j + 17) % 64) << 6) | ((j % 5) << 12)) as u16)
                    .collect()
            })
            .collect();
        let actual = model.evaluate(
            (0..size).map(|i| board(i + iteration)),
            moves.iter().map(|r| r.iter().copied()),
        )?;
        let mut features = Vec::new();
        let mut hints = Vec::new();
        for i in 0..size {
            let (b, black) = board(i + iteration);
            hints.push(model.hints.lookup(
                &crate::hints::material_key(&b.board, black),
                model.net.config.hints,
            ));
            encode_position(&mut features, &b, black).unwrap();
        }
        let unpadded = PolicyInputs::new(
            features,
            moves.iter().map(|r| r.iter().copied()),
            &model.device,
        )?
        .with_hints(hints)?;
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

#[test]
#[ignore = "requires CUDA and the baseline checkpoint"]
fn material_hints_survive_cuda_graph_replay() -> Result<(), Box<dyn std::error::Error>> {
    use crate::hints::{HintConfig, TableBuilder, material_key};
    use candle_nn::{VarBuilder, VarMap};
    let settings: toml::Value = toml::from_str(&std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../config/eval-model.toml"
    ))?)?;
    let baseline = InferenceModel::load(
        Path::new(settings["baseline_dir"].as_str().unwrap()),
        DeviceConfig::Cuda,
    )?;
    let mut config = baseline.net.config;
    config.hints = HintConfig::Material {
        min_games: 20.try_into()?,
        injection: crate::hints::Injection::EarlyAndValue,
        dropout: 0.,
    };
    let device = baseline.device;
    let vars = VarMap::new();
    let _ = Net::new(
        VarBuilder::from_varmap(&vars, config.inference_precision.dtype(), &device),
        config,
    )?;
    let tensors = vars
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(name, var)| (name.clone(), var.as_tensor().detach()))
        .collect();
    let net = Net::new(
        VarBuilder::from_tensors(tensors, config.inference_precision.dtype(), &device),
        config,
    )?;
    let mut builder = TableBuilder::default();
    let (b, black) = board(0);
    for game in 0..20 {
        builder.observe(0, game, material_key(&b.board, black), black, 1);
    }
    let model = InferenceModel {
        net,
        device,
        hints: builder.finish(),
        cuda: Default::default(),
    };
    for offset in [0, 1, 0] {
        for size in [8, 3] {
            let moves = vec![vec![0u16, 1, 2, 3]; size];
            let actual = model.evaluate(
                (0..size).map(|i| board(i + offset)),
                moves.iter().map(|m| m.iter().copied()),
            )?;
            let mut features = Vec::new();
            let mut hints = Vec::new();
            for i in 0..size {
                let (b, black) = board(i + offset);
                hints.push(
                    model
                        .hints
                        .lookup(&material_key(&b.board, black), config.hints),
                );
                encode_position(&mut features, &b, black).unwrap();
            }
            let inputs = PolicyInputs::new(
                features,
                moves.iter().map(|m| m.iter().copied()),
                &model.device,
            )?
            .with_hints(hints)?;
            let (p, v) = inputs.logits(&model.net)?;
            let expected = predictions(
                moves.iter().map(|m| m.iter().copied()),
                p.to_dtype(DType::F32)?.to_vec2()?,
                v.to_dtype(DType::F32)?.to_vec2()?,
            )?;
            for (actual, expected) in actual.iter().zip(expected) {
                assert_eq!(actual.value, expected.value);
                assert_eq!(actual.draw, expected.draw);
                assert_eq!(actual.priors, expected.priors);
            }
        }
    }
    Ok(())
}

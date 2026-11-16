use super::*;
use candle_core::Device;
use candle_nn::VarMap;

fn config() -> HintConfig {
    HintConfig::Material {
        min_games: NonZeroUsize::new(20).unwrap(),
        injection: crate::hints::Injection::Early,
        dropout: 0.,
    }
}

#[test]
fn table_counts_games_once_and_averages_symmetric_sides() {
    let key = [1; 10];
    let mut builder = TableBuilder::default();
    for game in 0..20 {
        for _ in 0..100 {
            builder.observe(0, game, key, false, 1);
        }
        builder.observe(0, game, key, true, -1);
    }
    let table = builder.finish();
    let features = table.lookup(&key, config());
    assert_eq!(features[..3], [11. / 23., 1. / 23., 11. / 23.]);
    assert!(features[3] >= 1.);
    assert_eq!(table.lookup(&[2; 10], config()), [0.; FEATURES]);
    let higher = HintConfig::Material {
        min_games: NonZeroUsize::new(21).unwrap(),
        injection: crate::hints::Injection::Early,
        dropout: 0.,
    };
    assert_eq!(table.lookup(&key, higher), [0.; FEATURES]);
    assert_eq!(table.lookup(&key, HintConfig::Disabled {}), [0.; FEATURES]);
}

#[test]
fn material_keys_swap_sides_and_ignore_piece_locations() {
    let mut board = [0; 32];
    board[0] = 1 | (8 << 4); // White pawn, black knight.
    let white = material_key(&board, false);
    let black = material_key(&board, true);
    assert_eq!(white[0], 1);
    assert_eq!(white[6], 1);
    assert_eq!(&white[..5], &black[5..]);
    board.swap(0, 31);
    assert_eq!(material_key(&board, false), white);
}

#[test]
fn table_round_trip_and_missing_checkpoint_data() -> Result<()> {
    let mut builder = TableBuilder::default();
    for game in 0..20 {
        builder.observe(0, game, [1; 10], false, 1);
    }
    let table = builder.finish();
    let mut tensors = std::collections::HashMap::new();
    table.save(&mut tensors)?;
    let restored = MaterialTable::load(&mut tensors)?;
    assert_eq!(
        table.lookup(&[1; 10], config()),
        restored.lookup(&[1; 10], config())
    );
    assert!(MaterialTable::load(&mut tensors).is_err());
    Ok(())
}

#[test]
fn embeddings_mask_missing_hints_and_both_receive_gradients() -> Result<()> {
    let vars = VarMap::new();
    let device = &Device::Cpu;
    let embedding = HintEmbedding::new(VarBuilder::from_varmap(&vars, DType::F32, device), 8)?;
    let missing = Tensor::zeros((2, FEATURES), DType::F32, device)?;
    assert_eq!(
        embedding
            .forward(&missing)?
            .abs()?
            .sum_all()?
            .to_scalar::<f32>()?,
        0.
    );
    let features = Tensor::from_slice(
        &[0.2f32, 0.3, 0.5, 1., 0.6, 0.2, 0.2, 5.],
        (2, FEATURES),
        device,
    )?;
    let output = embedding.forward(&features)?;
    let gradients = output.sqr()?.sum_all()?.backward()?;
    for (name, var) in vars.data().lock().unwrap().iter() {
        let gradient = gradients
            .get(var)
            .expect("hint parameters receive gradients");
        assert!(
            gradient.abs()?.sum_all()?.to_scalar::<f32>()? > 0.,
            "{name}"
        );
    }
    Ok(())
}

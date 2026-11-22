use chess_config::toml_config;

#[toml_config]
#[derive(Debug, serde::Serialize)]
struct Settings {
    #[range(min_exclusive = 0.0, max = 1.0)]
    value: f64,
    mode: Mode,
}

#[toml_config]
#[derive(Debug, serde::Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Mode {
    Enabled {
        #[range(min = -0.5, max_exclusive = 1.5)]
        value: f64,
    },
    Disabled {},
}

#[test]
fn ranges_validate_struct_and_enum_fields() -> Result<(), Box<dyn std::error::Error>> {
    let source = |value: &str, nested: &str| {
        format!("value = {value}\n[mode]\nkind = 'enabled'\nvalue = {nested}")
    };
    for (value, nested) in [("1", "-0.5"), ("0.5", "1.49")] {
        let config: Settings = toml::from_str(&source(value, nested))?;
        let serialized = toml::to_string(&config)?;
        let _: Settings = toml::from_str(&serialized)?;
    }
    for value in ["0", "-0.1", "1.01", "nan", "inf", "-inf"] {
        let error = toml::from_str::<Settings>(&source(value, "0")).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("value must be finite and satisfy")
        );
    }
    for nested in ["-0.51", "1.5", "nan", "inf", "-inf"] {
        assert!(toml::from_str::<Settings>(&source("0.5", nested)).is_err());
    }
    assert!(toml::from_str::<Settings>("value = 0.5\n[mode]\nkind = 'disabled'").is_ok());
    Ok(())
}

#[toml_config]
#[derive(Debug)]
struct Positive {
    #[range(min_exclusive = 0.0)]
    value: f64,
}

#[test]
fn one_sided_ranges_have_no_artificial_upper_bound() -> Result<(), Box<dyn std::error::Error>> {
    let config: Positive = toml::from_str(&format!("value = {:e}", f64::MAX))?;
    assert_eq!(config.value, f64::MAX);
    for value in ["0", "-1", "inf", "nan"] {
        assert!(toml::from_str::<Positive>(&format!("value = {value}")).is_err());
    }
    Ok(())
}

#[toml_config]
#[derive(Debug, serde::Serialize)]
struct SinglePrecision {
    #[range(min = 0.0)]
    value: f32,
}

#[test]
fn f32_ranges_reject_overflow_and_nonfinite_values() -> Result<(), Box<dyn std::error::Error>> {
    for raw in ["0", "0.5", "3.4028235e38"] {
        let config: SinglePrecision = toml::from_str(&format!("value = {raw}"))?;
        assert!(config.value.is_finite());
        let restored: SinglePrecision = toml::from_str(&toml::to_string(&config)?)?;
        assert_eq!(restored.value, config.value);
    }
    for raw in ["-0.1", "1e39", "inf", "-inf", "nan"] {
        assert!(toml::from_str::<SinglePrecision>(&format!("value = {raw}")).is_err());
    }
    Ok(())
}

#[test]
fn partial_configs_inherit_omitted_fields_and_validate_overrides()
-> Result<(), Box<dyn std::error::Error>> {
    let baseline: Settings = toml::from_str("value = 0.5\n[mode]\nkind = 'enabled'\nvalue = 1.0")?;
    let apply = |source: &str| {
        toml::from_str::<chess_config::PartialConfig<Settings>>(source)?.apply(&baseline)
    };
    let changed = apply("value = 0.75")?;
    assert_eq!(changed.value, 0.75);
    assert!(matches!(changed.mode, Mode::Enabled { value: 1.0 }));
    assert_eq!(apply("")?.value, 0.5);
    for invalid in ["unknown = true", "value = 'wrong type'", "value = -1.0"] {
        assert!(apply(invalid).is_err());
    }
    Ok(())
}

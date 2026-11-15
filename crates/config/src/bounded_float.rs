use serde::{Deserialize, Deserializer, Serialize, de::Error};

/// A finite `f32` with inclusive integer bounds, serialized as an ordinary number.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct BoundedFloat<const MIN: i32, const MAX: i32>(f32);

pub type UnitInterval = BoundedFloat<0, 1>;

impl<const MIN: i32, const MAX: i32> BoundedFloat<MIN, MAX> {
    pub fn new(value: f32) -> Result<Self, String> {
        Self::validate(f64::from(value))?;
        Ok(Self(value))
    }

    pub const fn get(self) -> f32 {
        self.0
    }

    fn validate(value: f64) -> Result<(), String> {
        if value.is_finite() && (f64::from(MIN)..=f64::from(MAX)).contains(&value) {
            Ok(())
        } else {
            Err(format!(
                "expected a finite number in [{MIN}, {MAX}], got {value}"
            ))
        }
    }
}

impl<'de, const MIN: i32, const MAX: i32> Deserialize<'de> for BoundedFloat<MIN, MAX> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = f64::deserialize(deserializer)?;
        // Validate before narrowing so out-of-range values cannot round into range.
        Self::validate(value).map_err(D::Error::custom)?;
        Self::new(value as f32).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Deserialize, Serialize)]
    struct Settings {
        value: UnitInterval,
    }

    #[test]
    fn toml_accepts_endpoints_and_round_trips_as_a_number() {
        for value in ["0", "0.0", "0.25", "1", "1.0"] {
            let settings: Settings = toml::from_str(&format!("value = {value}")).unwrap();
            assert_eq!(settings.value.get(), value.parse::<f32>().unwrap());
            let text = toml::to_string(&settings).unwrap();
            let restored: Settings = toml::from_str(&text).unwrap();
            assert_eq!(restored.value, settings.value);
        }
    }

    #[test]
    fn rejects_invalid_values_before_float_rounding() {
        for value in [
            "-0.1",
            "1.1",
            "1.000000001",
            "-1e-100",
            "nan",
            "inf",
            "-inf",
        ] {
            let error = toml::from_str::<Settings>(&format!("value = {value}")).unwrap_err();
            assert!(error.to_string().contains("finite number in [0, 1]"));
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
            assert!(UnitInterval::new(value).is_err());
        }
    }

    #[test]
    fn supports_other_bounds_and_rejects_reversed_bounds() {
        assert_eq!(BoundedFloat::<-2, 3>::new(-1.5).unwrap().get(), -1.5);
        assert!(BoundedFloat::<-2, 3>::new(3.1).is_err());
        assert!(BoundedFloat::<2, 1>::new(1.5).is_err());
    }
}

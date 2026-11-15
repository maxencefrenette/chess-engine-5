use std::marker::PhantomData;

use serde::{Deserialize, Deserializer, Serialize, de::Error};

pub trait Endpoint: Copy {
    const INCLUDED: bool;
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Inclusive;
impl Endpoint for Inclusive {
    const INCLUDED: bool = true;
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Exclusive;
impl Endpoint for Exclusive {
    const INCLUDED: bool = false;
}

/// A finite `f64` with integer bounds, serialized as an ordinary number.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct BoundedFloat<const MIN: i32, const MAX: i32, Lower: Endpoint, Upper: Endpoint>(
    f64,
    #[serde(skip)] PhantomData<(Lower, Upper)>,
);

impl<const MIN: i32, const MAX: i32, Lower: Endpoint, Upper: Endpoint>
    BoundedFloat<MIN, MAX, Lower, Upper>
{
    pub fn new(value: f64) -> Result<Self, String> {
        let above = value > f64::from(MIN) || (Lower::INCLUDED && value == f64::from(MIN));
        let below = value < f64::from(MAX) || (Upper::INCLUDED && value == f64::from(MAX));
        if value.is_finite() && above && below {
            Ok(Self(value, PhantomData))
        } else {
            let left = if Lower::INCLUDED { '[' } else { '(' };
            let right = if Upper::INCLUDED { ']' } else { ')' };
            Err(format!(
                "expected a finite number in {left}{MIN}, {MAX}{right}, got {value}"
            ))
        }
    }

    pub const fn get(self) -> f64 {
        self.0
    }
}

impl<const MIN: i32, const MAX: i32, Lower: Endpoint, Upper: Endpoint> TryFrom<f64>
    for BoundedFloat<MIN, MAX, Lower, Upper>
{
    type Error = String;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl<const MIN: i32, const MAX: i32, Lower: Endpoint, Upper: Endpoint> std::fmt::Display
    for BoundedFloat<MIN, MAX, Lower, Upper>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<'de, const MIN: i32, const MAX: i32, Lower: Endpoint, Upper: Endpoint> Deserialize<'de>
    for BoundedFloat<MIN, MAX, Lower, Upper>
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(f64::deserialize(deserializer)?).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Deserialize, Serialize)]
    struct Settings {
        value: BoundedFloat<0, 1, Inclusive, Inclusive>,
    }

    #[test]
    fn try_into_uses_the_destination_bounds() -> Result<(), String> {
        let value: BoundedFloat<0, 1, Exclusive, Inclusive> = 0.5.try_into()?;
        assert_eq!(value.get(), 0.5);
        for raw in [0.0, -0.1, 1.1, f64::NAN, f64::INFINITY] {
            let result: Result<BoundedFloat<0, 1, Exclusive, Inclusive>, _> = raw.try_into();
            assert!(result.is_err());
        }
        Ok(())
    }

    #[test]
    fn toml_accepts_endpoints_and_round_trips_as_a_number() {
        for value in ["0", "0.0", "0.25", "1", "1.0"] {
            let settings: Settings = toml::from_str(&format!("value = {value}")).unwrap();
            assert_eq!(settings.value.get(), value.parse::<f64>().unwrap());
            let text = toml::to_string(&settings).unwrap();
            let restored: Settings = toml::from_str(&text).unwrap();
            assert_eq!(restored.value, settings.value);
        }
    }

    #[test]
    fn rejects_nonfinite_and_out_of_range_values() {
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
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1] {
            assert!(BoundedFloat::<0, 1, Inclusive, Inclusive>::new(value).is_err());
        }
    }

    #[test]
    fn supports_other_bounds_and_rejects_reversed_bounds() {
        assert_eq!(
            BoundedFloat::<-2, 3, Inclusive, Inclusive>::new(-1.5)
                .unwrap()
                .get(),
            -1.5
        );
        assert!(BoundedFloat::<-2, 3, Inclusive, Inclusive>::new(3.1).is_err());
        assert!(BoundedFloat::<2, 1, Inclusive, Inclusive>::new(1.5).is_err());
    }
    #[test]
    fn endpoint_markers_control_each_boundary() {
        assert!(BoundedFloat::<0, 1, Exclusive, Inclusive>::new(0.).is_err());
        assert!(BoundedFloat::<0, 1, Exclusive, Inclusive>::new(1.).is_ok());
        assert!(BoundedFloat::<0, 1, Inclusive, Exclusive>::new(0.).is_ok());
        assert!(BoundedFloat::<0, 1, Inclusive, Exclusive>::new(1.).is_err());
        assert!(BoundedFloat::<1, 1, Inclusive, Inclusive>::new(1.).is_ok());
        assert!(BoundedFloat::<1, 1, Exclusive, Inclusive>::new(1.).is_err());
        assert!(BoundedFloat::<1, 1, Inclusive, Exclusive>::new(1.).is_err());
    }

    #[test]
    fn exclusive_intervals_deserialize_and_serialize_as_numbers() {
        #[derive(Debug, Deserialize, Serialize)]
        struct Settings {
            value: BoundedFloat<0, 1, Exclusive, Exclusive>,
        }
        for raw in ["0", "1", "nan", "inf", "-inf"] {
            assert!(toml::from_str::<Settings>(&format!("value = {raw}")).is_err());
        }
        let settings: Settings = toml::from_str("value = 0.123456789012345").unwrap();
        assert_eq!(settings.value.get(), 0.123456789012345);
        let serialized = toml::to_string(&settings).unwrap();
        let restored: Settings = toml::from_str(&serialized).unwrap();
        assert_eq!(settings.value, restored.value);
        assert_eq!(settings.value.to_string(), "0.123456789012345");
    }
}

//! Dedicated SVG charts for experiment reports.
mod style;

use std::{error::Error, ops::Range, path::Path};

use chess_config::TomlConfig;
use plotters::{
    prelude::*,
    style::text_anchor::{HPos, Pos, VPos},
};
use serde::Deserialize;
use style::{FONT, INK, LINE_WIDTH, color, frame};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
pub struct NumericNelo {
    pub x: f64,
    pub nelo: f64,
    pub lower: f64,
    pub upper: f64,
}

#[derive(Deserialize)]
pub struct CategoricalNelo {
    pub label: String,
    pub nelo: f64,
    pub lower: f64,
    pub upper: f64,
}

#[derive(Deserialize)]
pub struct NumericLoss {
    pub x: f64,
    pub loss: f64,
}

#[derive(Deserialize)]
pub struct CategoricalLoss {
    pub label: String,
    pub loss: f64,
}

pub fn nelo_numeric(path: &Path, x_label: &str, rows: &[NumericNelo]) -> Result<()> {
    let points = rows
        .iter()
        .map(|r| Point::nelo(r.x, r.nelo, r.lower, r.upper))
        .collect();
    draw(path, x_label, Metric::Nelo, Axis::Numeric, points)
}

pub fn nelo_categorical(path: &Path, x_label: &str, rows: &[CategoricalNelo]) -> Result<()> {
    let labels = rows.iter().map(|r| r.label.as_str()).collect();
    let points = rows
        .iter()
        .enumerate()
        .map(|(i, r)| Point::nelo(i as f64, r.nelo, r.lower, r.upper))
        .collect();
    draw(
        path,
        x_label,
        Metric::Nelo,
        Axis::Categories(labels),
        points,
    )
}

#[derive(Deserialize, TomlConfig)]
struct LossCalibration {
    loss_stddev: f64,
}

fn loss_margin() -> Result<f64> {
    let calibration = LossCalibration::from_repo_file("eval-model.toml")?;
    if !calibration.loss_stddev.is_finite() || calibration.loss_stddev < 0.0 {
        return Err("loss_stddev must be finite and nonnegative".into());
    }
    // Normal approximation for one training run at each point.
    Ok(1.96 * calibration.loss_stddev)
}

pub fn loss_numeric(path: &Path, x_label: &str, rows: &[NumericLoss]) -> Result<()> {
    let margin = loss_margin()?;
    let points = rows
        .iter()
        .map(|r| Point::loss(r.x, r.loss, margin))
        .collect();
    draw(path, x_label, Metric::Loss, Axis::Numeric, points)
}

pub fn loss_categorical(path: &Path, x_label: &str, rows: &[CategoricalLoss]) -> Result<()> {
    let margin = loss_margin()?;
    let labels = rows.iter().map(|r| r.label.as_str()).collect();
    let points = rows
        .iter()
        .enumerate()
        .map(|(i, r)| Point::loss(i as f64, r.loss, margin))
        .collect();
    draw(
        path,
        x_label,
        Metric::Loss,
        Axis::Categories(labels),
        points,
    )
}

struct Point {
    x: f64,
    y: f64,
    lower: f64,
    upper: f64,
}

impl Point {
    fn nelo(x: f64, y: f64, lower: f64, upper: f64) -> Self {
        Self { x, y, lower, upper }
    }

    fn loss(x: f64, y: f64, margin: f64) -> Self {
        Self {
            x,
            y,
            lower: y - margin,
            upper: y + margin,
        }
    }
}

enum Axis<'a> {
    Numeric,
    Categories(Vec<&'a str>),
}

#[derive(PartialEq)]
enum Metric {
    Loss,
    Nelo,
}

fn range(values: impl Iterator<Item = f64>) -> Result<Range<f64>> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for value in values {
        if !value.is_finite() {
            return Err("chart values must be finite".into());
        }
        min = min.min(value);
        max = max.max(value);
    }
    let padding = if min == max {
        min.abs().max(1.0) * 0.05
    } else {
        (max - min) * 0.05
    };
    let bounds = min - padding..max + padding;
    if !(bounds.end - bounds.start).is_finite() {
        return Err("chart values exceed the axis range".into());
    }
    Ok(bounds)
}

fn draw(
    path: &Path,
    x_label: &str,
    metric: Metric,
    axis: Axis,
    mut points: Vec<Point>,
) -> Result<()> {
    if points.is_empty() {
        return Err("chart needs at least one point".into());
    }
    for point in &points {
        if !(point.lower <= point.y && point.y <= point.upper) {
            return Err("expected lower <= estimate <= upper".into());
        }
    }
    let zero = (metric == Metric::Nelo).then_some(0.0);
    let y = range(points.iter().flat_map(|p| [p.lower, p.upper]).chain(zero))?;

    let numeric = matches!(axis, Axis::Numeric);
    let (x, width, ticks) = match axis {
        Axis::Numeric => {
            let x = range(points.iter().map(|p| p.x))?;
            points.sort_by(|a, b| a.x.total_cmp(&b.x));
            let mut ticks: Vec<_> = points.iter().map(|p| (p.x, p.x.to_string())).collect();
            ticks.dedup_by(|a, b| a.0 == b.0);
            // Fall back to evenly spaced numeric ticks when observed values would overlap.
            if ticks.array_windows().any(|[(a, first), (b, second)]| {
                (b - a) / (x.end - x.start) * 600.0
                    < (first.len() + second.len()) as f64 * 4.5 + 8.0
            }) {
                ticks.clear();
            }
            (x, 720, ticks)
        }
        Axis::Categories(labels) => {
            if labels.iter().any(|s| s.trim().is_empty()) {
                return Err("category labels must not be empty".into());
            }
            let longest = labels.iter().map(|s| s.chars().count()).max().unwrap_or(0);
            let width = u32::try_from((120 + labels.len() * (9 * longest + 24)).max(720))?;
            let x = -0.5..labels.len() as f64 - 0.5;
            let ticks = labels
                .into_iter()
                .enumerate()
                .map(|(i, s)| (i as f64, s.to_owned()))
                .collect();
            (x, width, ticks)
        }
    };

    let (name, y_label) = match metric {
        Metric::Nelo => ("nElo", "nElo (95% CI)"),
        Metric::Loss => ("Loss", "Loss (95% CI)"),
    };
    let root = SVGBackend::new(path, (width, 420)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{name} vs {x_label}"),
            (FONT, 20).into_font().color(&INK),
        )
        .margin(24)
        .x_label_area_size(48)
        .y_label_area_size(72)
        .build_cartesian_2d(x.clone(), y.clone())?;
    let mut mesh = chart.configure_mesh();
    if !ticks.is_empty() {
        mesh.x_labels(0);
    }
    mesh.x_desc(x_label)
        .y_desc(y_label)
        .label_style((FONT, 15).into_font().color(&INK))
        .axis_desc_style((FONT, 17).into_font().color(&INK))
        .axis_style(INK)
        .disable_mesh()
        .draw()?;
    for (value, label) in ticks {
        let (px, py) = chart.backend_coord(&(value, y.start));
        root.draw(&PathElement::new([(px, py), (px, py - 4)], INK))?;
        root.draw(&Text::new(
            label,
            (px, py + 8),
            (FONT, 15)
                .into_font()
                .color(&INK)
                .pos(Pos::new(HPos::Center, VPos::Top)),
        ))?;
    }
    if let Some(zero) = zero {
        chart.draw_series([PathElement::new(
            [(x.start, zero), (x.end, zero)],
            INK.mix(0.4),
        )])?;
    }
    if numeric {
        chart.draw_series(LineSeries::new(
            points.iter().map(|p| (p.x, p.y)),
            color(0, 1).stroke_width(LINE_WIDTH),
        ))?;
    }
    for (i, p) in points.iter().enumerate() {
        let ink = if numeric {
            color(0, 1)
        } else {
            color(i, points.len())
        };
        chart.draw_series([ErrorBar::new_vertical(
            p.x,
            p.lower,
            p.y,
            p.upper,
            ink.filled().stroke_width(LINE_WIDTH),
            10,
        )])?;
    }
    frame(&root, chart.plotting_area())?;
    root.present()?;
    Ok(())
}

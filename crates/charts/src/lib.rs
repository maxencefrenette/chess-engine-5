//! Shared SVG charts for experiment reports.
use std::{error::Error, ops::Range, path::Path};

use plotters::{coord::Shift, prelude::*};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

const FONT: &str = "sans-serif";
const INK: RGBColor = RGBColor(42, 48, 58);
const GRID: RGBColor = RGBColor(230, 233, 238);
const COLORS: [RGBColor; 6] = [
    RGBColor(0, 114, 178),
    RGBColor(213, 94, 0),
    RGBColor(0, 158, 115),
    RGBColor(204, 121, 167),
    RGBColor(86, 180, 233),
    RGBColor(145, 105, 20),
];

pub struct Series {
    pub label: String,
    pub points: Vec<(f64, f64)>,
}

pub struct Estimate {
    pub label: String,
    pub nelo: f64,
    pub lower: f64,
    pub upper: f64,
}

pub struct SweepPoint {
    pub x: f64,
    pub nelo: f64,
    pub lower: f64,
    pub upper: f64,
}

fn valid_estimate(nelo: f64, lower: f64, upper: f64) -> bool {
    [nelo, lower, upper].iter().all(|v| v.is_finite()) && lower <= nelo && nelo <= upper
}

fn canvas(path: &Path, height: u32) -> Result<DrawingArea<SVGBackend<'_>, Shift>> {
    let root = SVGBackend::new(path, (960, height)).into_drawing_area();
    root.fill(&WHITE)?;
    Ok(root)
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
    if min > max {
        return Err("chart needs at least one point".into());
    }
    let padding = if min == max {
        min.abs().max(1.0) * 0.05
    } else {
        (max - min) * 0.05
    };
    Ok(min - padding..max + padding)
}

/// Draw unsmoothed curves with a shared palette, typography, and legend.
pub fn lines(
    path: &Path,
    title: &str,
    x_label: &str,
    y_label: &str,
    series: &[Series],
) -> Result<()> {
    if series.is_empty() || series.iter().any(|s| s.points.is_empty()) {
        return Err("each series needs at least one point".into());
    }
    let x = range(series.iter().flat_map(|s| s.points.iter().map(|p| p.0)))?;
    let y = range(series.iter().flat_map(|s| s.points.iter().map(|p| p.1)))?;
    let root = canvas(path, 560)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, (FONT, 24).into_font().color(&INK))
        .margin(24)
        .x_label_area_size(48)
        .y_label_area_size(72)
        .build_cartesian_2d(x, y)?;
    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .label_style((FONT, 14).into_font().color(&INK))
        .axis_desc_style((FONT, 16).into_font().color(&INK))
        .axis_style(INK)
        .light_line_style(TRANSPARENT)
        .bold_line_style(GRID)
        .draw()?;
    for (i, series) in series.iter().enumerate() {
        let color = COLORS[i % COLORS.len()];
        chart
            .draw_series(LineSeries::new(
                series.points.iter().copied(),
                color.stroke_width(2),
            ))?
            .label(&series.label)
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 24, y)], color.stroke_width(2)));
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.95))
        .border_style(GRID)
        .label_font((FONT, 14))
        .draw()?;
    root.present()?;
    Ok(())
}

/// Plot supplied nElo estimates and 95% CIs; statistical calculations stay outside this crate.
pub fn nelo(path: &Path, title: &str, estimates: &[Estimate]) -> Result<()> {
    if estimates.is_empty()
        || estimates
            .iter()
            .any(|e| !valid_estimate(e.nelo, e.lower, e.upper))
    {
        return Err("expected finite estimates with lower <= nelo <= upper".into());
    }
    let x = range(std::iter::once(0.0).chain(estimates.iter().flat_map(|e| [e.lower, e.upper])))?;
    let root = canvas(path, (180 + estimates.len() as u32 * 64).max(360))?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, (FONT, 24).into_font().color(&INK))
        .margin(24)
        .x_label_area_size(48)
        .y_label_area_size(180)
        .build_cartesian_2d(x, -0.5..estimates.len() as f64 - 0.5)?;
    chart
        .configure_mesh()
        .x_desc("nElo (95% CI)")
        .y_labels(0)
        .label_style((FONT, 14).into_font().color(&INK))
        .axis_desc_style((FONT, 16).into_font().color(&INK))
        .axis_style(INK)
        .light_line_style(TRANSPARENT)
        .bold_line_style(GRID)
        .draw()?;
    chart.draw_series(std::iter::once(PathElement::new(
        [(0.0, -0.5), (0.0, estimates.len() as f64 - 0.5)],
        INK.mix(0.4),
    )))?;
    for (i, estimate) in estimates.iter().enumerate() {
        let y = (estimates.len() - i - 1) as f64;
        let color = COLORS[i % COLORS.len()];
        chart.draw_series([
            PathElement::new(
                [(estimate.lower, y), (estimate.upper, y)],
                color.stroke_width(2),
            ),
            PathElement::new(
                [(estimate.lower, y - 0.1), (estimate.lower, y + 0.1)],
                color.stroke_width(2),
            ),
            PathElement::new(
                [(estimate.upper, y - 0.1), (estimate.upper, y + 0.1)],
                color.stroke_width(2),
            ),
        ])?;
        chart.draw_series(std::iter::once(Circle::new(
            (estimate.nelo, y),
            5,
            color.filled(),
        )))?;
        let (_, pixel_y) = chart.backend_coord(&(estimate.nelo, y));
        root.draw(&Text::new(
            estimate.label.as_str(),
            (24, pixel_y - 7),
            (FONT, 15).into_font().color(&INK),
        ))?;
    }
    root.present()?;
    Ok(())
}

/// Numeric sweep with supplied 95% CI endpoints. Connect points in x order without fitting.
pub fn sweep(path: &Path, title: &str, x_label: &str, points: &[SweepPoint]) -> Result<()> {
    if points.is_empty()
        || points
            .iter()
            .any(|p| !p.x.is_finite() || !valid_estimate(p.nelo, p.lower, p.upper))
    {
        return Err("expected finite sweep points with lower <= nelo <= upper".into());
    }
    let mut points: Vec<_> = points.iter().collect();
    points.sort_by(|a, b| a.x.total_cmp(&b.x));
    let x = range(points.iter().map(|p| p.x))?;
    let y = range(std::iter::once(0.0).chain(points.iter().flat_map(|p| [p.lower, p.upper])))?;
    let root = canvas(path, 560)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, (FONT, 24).into_font().color(&INK))
        .margin(24)
        .x_label_area_size(48)
        .y_label_area_size(72)
        .build_cartesian_2d(x.clone(), y)?;
    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc("nElo (95% CI)")
        .label_style((FONT, 14).into_font().color(&INK))
        .axis_desc_style((FONT, 16).into_font().color(&INK))
        .axis_style(INK)
        .light_line_style(TRANSPARENT)
        .bold_line_style(GRID)
        .draw()?;
    chart.draw_series(std::iter::once(PathElement::new(
        [(x.start, 0.0), (x.end, 0.0)],
        INK.mix(0.4),
    )))?;
    let color = COLORS[0];
    chart.draw_series(LineSeries::new(
        points.iter().map(|p| (p.x, p.nelo)),
        color.mix(0.5).stroke_width(2),
    ))?;
    chart.draw_series(points.iter().map(|p| {
        ErrorBar::new_vertical(
            p.x,
            p.lower,
            p.nelo,
            p.upper,
            color.filled().stroke_width(2),
            10,
        )
    }))?;
    root.present()?;
    Ok(())
}

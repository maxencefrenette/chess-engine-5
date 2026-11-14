use std::path::Path;

use plotters::{
    prelude::*,
    style::text_anchor::{HPos, Pos, VPos},
};

use crate::{Result, Series, range};

use crate::style::{FONT, INK, LEGEND_BORDER, LINE_WIDTH, MARKER_RADIUS, color, frame};

pub struct LinePanel {
    pub title: String,
    pub series: Vec<Series>,
}

/// Compact boxed panels with thin lines, solid markers, and one inset legend.
pub fn line_panels(path: &Path, x_label: &str, y_label: &str, panels: &[LinePanel]) -> Result<()> {
    if panels.is_empty()
        || panels.iter().any(|p| {
            p.series.is_empty()
                || p.series.iter().any(|s| s.points.is_empty())
                || p.series
                    .iter()
                    .map(|s| &s.label)
                    .ne(panels[0].series.iter().map(|s| &s.label))
        })
    {
        return Err("panels need nonempty series with identical ordered labels".into());
    }
    let width = 420 * panels.len() as u32;
    let root = SVGBackend::new(path, (width, 300)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut ticks: Vec<_> = panels
        .iter()
        .flat_map(|p| &p.series)
        .flat_map(|s| s.points.iter().map(|p| p.0))
        .collect();
    let x = range(ticks.iter().copied())?;
    ticks.sort_by(f64::total_cmp);
    ticks.dedup();
    for (index, (area, panel)) in root
        .split_evenly((1, panels.len()))
        .iter()
        .zip(panels)
        .enumerate()
    {
        let y = range(
            panel
                .series
                .iter()
                .flat_map(|s| s.points.iter().map(|p| p.1)),
        )?;
        let lower = y.start;
        let mut chart = ChartBuilder::on(area)
            .caption(&panel.title, (FONT, 19).into_font())
            .margin_top(10)
            .margin_left(12)
            .margin_right(18)
            .margin_bottom(8)
            .x_label_area_size(48)
            .y_label_area_size(64)
            .build_cartesian_2d(x.clone(), y)?;
        chart
            .configure_mesh()
            .x_desc(x_label)
            .y_desc(y_label)
            .x_labels(0)
            .y_labels(5)
            .y_label_formatter(&|value| format!("{value:.3}"))
            .disable_mesh()
            .label_style((FONT, 15).into_font())
            .axis_desc_style((FONT, 17).into_font())
            .axis_style(INK.stroke_width(LINE_WIDTH))
            .light_line_style(TRANSPARENT)
            .draw()?;
        for &tick in &ticks {
            let (px, py) = chart.backend_coord(&(tick, lower));
            root.draw(&PathElement::new([(px, py), (px, py - 4)], INK.mix(0.7)))?;
            root.draw(&Text::new(
                format!("{tick}"),
                (px, py + 8),
                (FONT, 15)
                    .into_font()
                    .color(&INK)
                    .pos(Pos::new(HPos::Center, VPos::Top)),
            ))?;
        }
        for (i, series) in panel.series.iter().enumerate() {
            let color = color(i, panel.series.len());
            let mut points = series.points.clone();
            points.sort_by(|a, b| a.0.total_cmp(&b.0));
            chart
                .draw_series(LineSeries::new(
                    points.iter().copied(),
                    color.stroke_width(LINE_WIDTH),
                ))?
                .label(&series.label)
                .legend(move |(x, y)| {
                    PathElement::new([(x, y), (x + 22, y)], color.stroke_width(LINE_WIDTH))
                });
            chart.draw_series(
                points
                    .iter()
                    .map(|&p| Circle::new(p, MARKER_RADIUS, color.filled())),
            )?;
        }
        if index + 1 == panels.len() {
            chart
                .configure_series_labels()
                .position(SeriesLabelPosition::UpperRight)
                .background_style(WHITE)
                .border_style(LEGEND_BORDER)
                .label_font((FONT, 15))
                .draw()?;
        }
        frame(&root, chart.plotting_area())?;
    }
    root.present()?;
    Ok(())
}

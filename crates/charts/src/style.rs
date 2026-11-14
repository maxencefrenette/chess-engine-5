use plotters::{coord::Shift, prelude::*};

use crate::Result;

pub(crate) const FONT: &str = "serif";
pub(crate) const INK: RGBColor = BLACK;
pub(crate) const LEGEND_BORDER: RGBColor = RGBColor(210, 210, 210);
pub(crate) const LINE_WIDTH: u32 = 1;
pub(crate) const MARKER_RADIUS: i32 = 3;

/// Spread ordered series across the purple–pink–yellow palette.
pub(crate) fn color(index: usize, count: usize) -> RGBColor {
    let anchors = [
        RGBColor(106, 0, 168),
        RGBColor(219, 65, 126),
        RGBColor(247, 184, 38),
    ];
    let position = 2.0 * index as f64 / count.saturating_sub(1).max(1) as f64;
    let segment = (position.floor() as usize).min(1);
    let fraction = position - segment as f64;
    let a = anchors[segment];
    let b = anchors[segment + 1];
    let mix =
        |a: u8, b: u8| (f64::from(a) + (f64::from(b) - f64::from(a)) * fraction).round() as u8;
    RGBColor(mix(a.0, b.0), mix(a.1, b.1), mix(a.2, b.2))
}

pub(crate) fn frame<DB: DrawingBackend, CT: CoordTranslate>(
    root: &DrawingArea<DB, Shift>,
    plot: &DrawingArea<DB, CT>,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let (x, y) = plot.get_pixel_range();
    root.draw(&Rectangle::new(
        [(x.start, y.start), (x.end - 1, y.end - 1)],
        INK.stroke_width(LINE_WIDTH),
    ))?;
    Ok(())
}

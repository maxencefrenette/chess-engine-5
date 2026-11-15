use super::*;

#[test]
#[ignore = "requires downloaded ChessTB tables on the HDD"]
fn probes_wdl_and_root_dtz() -> io::Result<()> {
    let mut tables = Tables::open(Path::new("/data/chess/tablebases/chesstb-5"))?;
    let fen = c"7k/8/5KQ1/8/8/8/8/8 w - - 99 1";
    assert_eq!(tables.wdl(fen, 64, 99)?, Wdl::Win);
    let moves = tables.root_dtz(fen, 64, 99, true, true)?;
    let mate = moves.iter().find(|m| m.from == 46 && m.to == 54).unwrap();
    assert_eq!(mate.dtz, 1);
    assert_eq!(mate.promotion, 0);
    Ok(())
}

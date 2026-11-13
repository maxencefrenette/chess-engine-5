use std::{
    fs, io,
    path::Path,
    process::{Command, Stdio},
};

use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

pub fn load(name: &str, pairs: usize, seed: u64) -> io::Result<Vec<String>> {
    let text = cached_book(Path::new("/data/chess/chess-engine-5/opening-books"), name)?;
    select(&text, pairs, seed)
}

fn select(text: &str, pairs: usize, seed: u64) -> io::Result<Vec<String>> {
    let mut lines: Vec<_> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    if lines.is_empty() {
        return Err(io::Error::other("opening book is empty"));
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(u64::MAX);
    lines.shuffle(&mut rng);
    Ok(lines
        .into_iter()
        .cycle()
        .take(pairs)
        .map(str::to_owned)
        .collect())
}

fn cached_book(directory: &Path, name: &str) -> io::Result<String> {
    if !name
        .bytes()
        .next()
        .is_some_and(|c| c.is_ascii_alphanumeric())
        || !name.ends_with(".epd")
        || !name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"._-+".contains(&c))
    {
        return Err(io::Error::other("opening book must be an EPD filename"));
    }
    let path = directory.join(name);
    match fs::read_to_string(&path) {
        Ok(text) => return Ok(text),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    fs::create_dir_all(directory)?;
    let temporary = tempfile::tempdir_in(directory)?;
    let archive = temporary.path().join("book.zip");
    let extracted = temporary.path().join("book.epd");
    let url =
        format!("https://raw.githubusercontent.com/official-stockfish/books/master/{name}.zip");
    eprintln!("Downloading opening book {name} to {}", path.display());
    let status = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "2",
            "--connect-timeout",
            "15",
            "--max-time",
            "300",
            "--output",
        ])
        .arg(&archive)
        .arg(&url)
        .status()?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "opening book download failed: {url}"
        )));
    }
    let status = Command::new("unzip")
        .arg("-p")
        .arg(&archive)
        .arg(name)
        .stdout(Stdio::from(fs::File::create(&extracted)?))
        .status()?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "could not extract opening book {name}"
        )));
    }
    let text = fs::read_to_string(&extracted)?;
    if text.trim().is_empty() {
        return Err(io::Error::other("opening book is empty"));
    }
    fs::rename(extracted, path)?;
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_is_seeded_and_exhausts_book_before_repeating() {
        let text = "first\nsecond\nthird\nfourth\n\n";
        let selected = select(text, 9, 42).unwrap();
        assert_eq!(selected, select(text, 9, 42).unwrap());
        assert_ne!(selected, select(text, 9, 43).unwrap());
        assert_eq!(
            selected[..4]
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            4
        );
        assert_eq!(selected[..4], selected[4..8]);
        assert_eq!(selected[0], selected[8]);
        assert!(select("\n", 1, 42).is_err());
    }

    #[test]
    fn cached_book_is_read_without_downloading() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("local.epd"), "cached contents\n").unwrap();
        assert_eq!(
            cached_book(directory.path(), "local.epd").unwrap(),
            "cached contents\n"
        );
        for name in ["../book.epd", "/book.epd", "book.pgn", "book?.epd"] {
            assert!(cached_book(directory.path(), name).is_err());
        }
    }
}

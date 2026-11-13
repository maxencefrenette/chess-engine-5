use super::Rule;

pub(super) const RULES: &[Rule] = &[Rule {
    name: "game_pair_terminology",
    pattern: r"(?i)\bcolor-swapped\s+(?:game\s+)?pairs?\b",
    help: "use ‘game pair’ or ‘game pairs’",
    reason: "swapping colors is implied by a game pair",
}];

#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;

    #[test]
    fn detects_redundant_pair_wording() {
        let regex = Regex::new(RULES[0].pattern).unwrap();
        for text in [
            "color-swapped pair",
            "color-swapped pairs",
            "COLOR-SWAPPED GAME PAIR",
            "color-swapped game pairs",
            "color-swapped\ngame pair",
        ] {
            assert!(regex.is_match(text), "{text}");
        }
        for text in [
            "game pair",
            "game pairs",
            "colors are swapped",
            "color-swapped games",
        ] {
            assert!(!regex.is_match(text), "{text}");
        }
    }
}

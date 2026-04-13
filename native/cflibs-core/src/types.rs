/// Score result for a single element's comb matching.
///
/// Tracks the number of matched/expected lines, precision, recall, F1 score,
/// and whether the element passes the acceptance thresholds.
#[derive(Debug, Clone)]
pub struct CombScoreResult {
    pub element: String,
    pub matched_lines: usize,
    pub expected_lines: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub missing_fraction: f64,
    pub passes: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comb_score_result_creation() {
        let score = CombScoreResult {
            element: "Fe".to_string(),
            matched_lines: 5,
            expected_lines: 10,
            precision: 0.25,
            recall: 0.5,
            f1_score: 0.333,
            missing_fraction: 0.5,
            passes: true,
        };
        assert_eq!(score.element, "Fe");
        assert_eq!(score.matched_lines, 5);
        assert_eq!(score.expected_lines, 10);
        assert!(score.passes);
    }

    #[test]
    fn test_comb_score_result_clone() {
        let score = CombScoreResult {
            element: "Cu".to_string(),
            matched_lines: 3,
            expected_lines: 8,
            precision: 0.15,
            recall: 0.375,
            f1_score: 0.214,
            missing_fraction: 0.625,
            passes: false,
        };
        let cloned = score.clone();
        assert_eq!(cloned.element, "Cu");
        assert_eq!(cloned.matched_lines, 3);
        assert!(!cloned.passes);
    }
}

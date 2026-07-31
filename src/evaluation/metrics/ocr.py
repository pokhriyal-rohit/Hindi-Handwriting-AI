"""
OCR Evaluation Metrics
Computes Character Error Rate (CER) and Word Error Rate (WER).
"""

from typing import List, Dict, Union

def edit_distance(ref: List[str], hyp: List[str]) -> int:
    """
    Computes Levenshtein distance between two sequences (can be list of chars or list of words).
    """
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,    # deletion
                    dp[i][j - 1] + 1,    # insertion
                    dp[i - 1][j - 1] + 1 # substitution
                )
    return dp[m][n]


class OCRMetrics:
    @staticmethod
    def compute_cer(reference: str, hypothesis: str) -> float:
        """
        Character Error Rate (CER) = EditDistance(ref_chars, hyp_chars) / len(ref_chars)
        """
        if not reference:
            return 1.0 if hypothesis else 0.0
            
        ref_chars = list(reference)
        hyp_chars = list(hypothesis)
        distance = edit_distance(ref_chars, hyp_chars)
        return distance / len(ref_chars)

    @staticmethod
    def compute_wer(reference: str, hypothesis: str) -> float:
        """
        Word Error Rate (WER) = EditDistance(ref_words, hyp_words) / len(ref_words)
        """
        if not reference:
            return 1.0 if hypothesis else 0.0
            
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        distance = edit_distance(ref_words, hyp_words)
        
        if not ref_words:
            return 1.0 if hyp_words else 0.0
            
        return distance / len(ref_words)

    @staticmethod
    def evaluate(reference: str, hypothesis: str) -> Dict[str, float]:
        return {
            "cer": OCRMetrics.compute_cer(reference, hypothesis),
            "wer": OCRMetrics.compute_wer(reference, hypothesis)
        }

import unittest
import torch
from src.tokenizers.devanagari import DevanagariTokenizer
from src.models.ocr.crnn import CRNN
from src.models.ocr.registry import build_ocr_model
from src.evaluation.metrics.ocr import OCRMetrics

class TestOCRModule(unittest.TestCase):
    def test_tokenizer(self):
        tokenizer = DevanagariTokenizer()
        texts = ["अनाथों", "आग", "पानी"]
        tokenizer.build_vocab(texts)
        
        self.assertGreater(tokenizer.vocab_size, 2)
        
        encoded = tokenizer.encode("आग")
        self.assertEqual(len(encoded), 2)
        
        decoded = tokenizer.decode(encoded, remove_repeats=True)
        self.assertEqual(decoded, "आग")
        
        # Test CTC decode collapse
        seq = [2, 2, 0, 2, 3, 3]
        tokenizer.char_to_idx['A'] = 2
        tokenizer.idx_to_char[2] = 'A'
        tokenizer.char_to_idx['B'] = 3
        tokenizer.idx_to_char[3] = 'B'
        
        decoded_ctc = tokenizer.decode(seq, remove_repeats=True)
        self.assertEqual(decoded_ctc, "AAB")
        
    def test_crnn_forward(self):
        config = {"img_channels": 1, "hidden_size": 128}
        vocab_size = 10
        model = build_ocr_model("crnn_baseline", vocab_size, config)
        
        batch_size = 2
        img_height = 32
        img_width = 128
        
        images = torch.randn(batch_size, 1, img_height, img_width)
        preds = model(images)
        
        self.assertEqual(preds.size(0), batch_size)
        self.assertEqual(preds.size(2), vocab_size)
        
        expected_t = (img_width // 4) - 1
        self.assertEqual(preds.size(1), expected_t)
        
        input_lengths = torch.tensor([img_width, img_width])
        out_lengths = model.get_output_length(input_lengths)
        self.assertEqual(out_lengths[0].item(), expected_t)

    def test_ocr_metrics(self):
        target = "अनाथों"
        pred1 = "अनाथों"
        pred2 = "अनाथ"
        
        self.assertEqual(OCRMetrics.compute_cer(target, pred1), 0.0)
        self.assertGreater(OCRMetrics.compute_cer(target, pred2), 0.0)
        
        target_words = "hello world"
        pred_words = "hello word"
        self.assertEqual(OCRMetrics.compute_wer(target_words, target_words), 0.0)
        self.assertEqual(OCRMetrics.compute_wer(target_words, pred_words), 0.5)

if __name__ == '__main__':
    unittest.main()

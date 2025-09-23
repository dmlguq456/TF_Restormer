import torch
import numpy as np
try:
    # Import from local utils modules
    from utils.util_speechbleu import SpeechBLEU
    from utils.util_speechbertscore import SpeechBERTScore  
    from utils.util_speechtokendistance import SpeechTokenDistance
    DISCRETE_SPEECH_METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Speech metrics not available. Error: {e}")
    print("SpeechBLEU, BERTScore, and TokenDistance metrics will be disabled.")
    DISCRETE_SPEECH_METRICS_AVAILABLE = False
    # Dummy classes for when the modules are not available
    class SpeechBLEU:
        pass
    class SpeechBERTScore:
        pass
    class SpeechTokenDistance:
        pass


class SpeechMetricCalculator:
    def __init__(self, metric_type: str, sr: int = 16000, device: str = "cuda"):
        """
        Args:
            metric_type (str): 'bleu', 'bertscore', 'tokendistance'
            **kwargs: 각 메트릭 클래스에 전달할 추가 인자.
                      (예: model_type, layer, vocab, n_ngram 등)
        """
        if not DISCRETE_SPEECH_METRICS_AVAILABLE:
            raise ImportError("discrete_speech_metrics is not available. Please install it to use this feature.")

        self.metric_type = metric_type.lower()

        common_args = {'sr': sr, 'device': device}

        if self.metric_type == 'bleu':
            bleu_args = {'model_type': "hubert-base", 'vocab': 200, 'layer': 11, 'n_ngram': 2}
            self.calculator = SpeechBLEU(**common_args, **bleu_args)
        elif self.metric_type == 'bertscore':
            bert_args = {'model_type': "wavlm-large", 'layer': 14}
            self.calculator = SpeechBERTScore(**common_args, **bert_args)
        elif self.metric_type == 'tokendistance':
            dist_args = {'model_type': "hubert-base", 'vocab': 200, 'layer': 6, 'distance_type': "jaro-winkler"}
            self.calculator = SpeechTokenDistance(**common_args, **dist_args)
        else:
            raise ValueError(f"Unsupported metric_type: {metric_type}. "
                             f"Choose from 'bleu', 'bertscore', 'tokendistance'.")

    def score(self, ref_wav: torch.Tensor, gen_wav: torch.Tensor):
        """
        Returns:
            메트릭 계산 결과. (bleu: float, bertscore: tuple, tokendistance: float)
        """
        if ref_wav.ndim > 1:
            ref_wav = ref_wav.squeeze()
        if gen_wav.ndim > 1:
            gen_wav = gen_wav.squeeze()
        
        ref_wav_np = ref_wav.detach().cpu().numpy()
        gen_wav_np = gen_wav.detach().cpu().numpy()

        return self.calculator.score(ref_wav_np, gen_wav_np)

if __name__ == '__main__':
    # --- 클래스 사용 예제 ---

    # 예제 텐서 생성 (16kHz에서 약 1초 분량)
    ref_wav_tensor = torch.randn(16000)
    gen_wav_tensor = torch.randn(16000)

    # 1. SpeechBLEU 계산
    bleu_scorer = SpeechMetricCalculator(metric_type='bleu', use_gpu=True)
    bleu_score = bleu_scorer.score(ref_wav_tensor, gen_wav_tensor)
    print(f"SpeechBLEU Score: {bleu_score:.4f}")

    # 2. SpeechBERTScore 계산
    bert_scorer = SpeechMetricCalculator(metric_type='bertscore', use_gpu=True)
    precision, recall, f1 = bert_scorer.score(ref_wav_tensor, gen_wav_tensor)
    print(f"SpeechBERTScore: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

    # 3. SpeechTokenDistance 계산
    token_dist_scorer = SpeechMetricCalculator(metric_type='tokendistance', use_gpu=True)
    distance = token_dist_scorer.score(ref_wav_tensor, gen_wav_tensor)
    print(f"SpeechTokenDistance: {distance:.4f}")
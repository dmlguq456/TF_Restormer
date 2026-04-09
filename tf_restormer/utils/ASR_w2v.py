import re
import torch as th
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from typing import List, Dict, Optional

class SimpleCTCASR:
    """
    - 입력: torch.Tensor (T,) 또는 (C, T)  # 16 kHz 가정, 배치 없음
    - __call__(wav) -> str (단어 또는 음소 시퀀스)
    - get_error_counts(src_wav, out_wav) -> (err, ref_units) # 누적용 (분자, 분모)
    """
    def __init__(
        self,
        metric_type: str = "wer",
        device: Optional[str] = None,
        phoneme_set: str = "arpabet",  # 'arpabet' | 'timit-39'  (PhER 전용 옵션)
        collapse_repeats: bool = False # 동일 음소 연속 중복 제거(선택)
    ):
        """
        metric_type에 따라 WER 또는 PhER 계산기를 초기화합니다.
        - 'wer': Word Error Rate (jonatasgrosman/wav2vec2-large-xlsr-53-english)
        - 'pher': Phoneme Error Rate (mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme)
        """
        self.metric_type = metric_type.lower()
        self.phoneme_set = phoneme_set
        self.collapse_repeats = collapse_repeats

        if self.metric_type == "wer":
            model_id = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        elif self.metric_type == "pher":
            model_id = "mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme"
        else:
            raise ValueError(f"지원하지 않는 metric_type: {metric_type}. 'wer' 또는 'pher'를 사용하세요.")

        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)

        if device is None:
            device = "cuda" if th.cuda.is_available() else ("mps" if th.backends.mps.is_available() else "cpu")
        self.device = th.device(device)
        self.model.to(self.device).eval()

        # ---- TIMIT-39 축소 매핑 (소문자 기준) ----
        # 참고: 61→39 표준 매핑에서 자주 등장하는 축약만 우선 반영. 필요 시 추가해도 됨.
        self._timit39_map: Dict[str, str] = {
            # Schwa/weak vowels
            "ax":"ah", "ix":"ih", "ax-h":"ah", "axr":"er",
            # Syllabic consonants
            "em":"m", "en":"n", "eng":"ng", "el":"l",
            # u variants
            "ux":"uw",
            # Flap
            "dx":"dx",  # 보존(혹은 't'로 매핑하고 싶으면 't'로 바꿔도 됨)
            # Others: 대부분 자기 자신으로 유지
        }
        # 침묵/노이즈 토큰(있다면 제거)
        self._silence_tokens = {"sil", "sp", "spn", "nsn", "pau"}

    @th.inference_mode()
    def __call__(self, wav: th.Tensor) -> str:
        # (C, T) -> mono, (T,) 그대로. 16 kHz 전제.
        if wav.ndim == 2:
            wav = wav.mean(dim=0)
        assert wav.ndim == 1, "input must be 1D or 2D (C, T) tensor"

        # int -> float, 범위 [-1, 1]
        if wav.dtype == th.int16:
            wav = wav.to(th.float32) / 32768.0
        elif wav.dtype == th.int32:
            wav = wav.to(th.float32) / 2147483648.0
        else:
            wav = wav.to(th.float32)
        wav = wav.clamp_(-1.0, 1.0)

        enc = self.processor(
            wav.detach().cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=False
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        logits = self.model(
            enc["input_values"],
            attention_mask=enc["attention_mask"] if "attention_mask" in enc else None
        ).logits
        ids = logits.argmax(dim=-1)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text

    # ---------- 텍스트 정규화 ----------
    @staticmethod
    def _normalize_wer(s: str) -> str:
        # 소문자 + 구두점 제거(아포스트로피 보존) + 공백 정리
        s = s.lower()
        s = re.sub(r"[^a-z0-9'\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _normalize_pher(self, s: str) -> str:
        # 1) 소문자
        s = s.lower()
        # 2) 강세 숫자 제거 (aa0/eh1/er2 등 → aa/eh/er)
        s = re.sub(r"\b([a-z]{2,})([0-2])\b", r"\1", s)
        # 3) 비알파벳 제거(기호/숫자 삭제). 단어 경계만 유지.
        s = re.sub(r"[^a-z\s]", " ", s)
        # 4) 공백 정리
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _normalize_text(self, s: str, remove_punct: Optional[bool]) -> str:
        # remove_punct가 None이면: WER=True / PhER=False의 관례 적용
        if self.metric_type == "wer":
            if remove_punct is None:
                remove_punct = True
            return self._normalize_wer(s) if remove_punct else s.lower().strip()
        else:  # 'pher'
            # PhER은 강세/기호 정리 규칙 고정
            return self._normalize_pher(s)

    # ---------- Phoneme 매핑/후처리 ----------
    def _map_phoneme(self, p: str) -> str:
        if self.phoneme_set == "timit-39":
            return self._timit39_map.get(p, p)
        return p

    def _postprocess_phonemes(self, toks: List[str]) -> List[str]:
        # 침묵 제거 + 선택적 중복축소 + timit-39 매핑
        out = []
        prev = None
        for t in toks:
            if t in self._silence_tokens or not t:  # 침묵/빈 토큰 제거
                continue
            t2 = self._map_phoneme(t)
            if not t2:
                continue
            if self.collapse_repeats:
                if prev is not None and t2 == prev:
                    continue
            out.append(t2)
            prev = t2
        return out

    # ---------- 편집거리 ----------
    @staticmethod
    def _edit_distance(ref_units: List[str], hyp_units: List[str]) -> int:
        R, H = len(ref_units), len(hyp_units)
        dp = [[0]*(H+1) for _ in range(R+1)]
        for i in range(R+1): dp[i][0] = i
        for j in range(H+1): dp[0][j] = j
        for i in range(1, R+1):
            ri = ref_units[i-1]
            for j in range(1, H+1):
                hj = hyp_units[j-1]
                cost = 0 if ri == hj else 1
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
        return dp[R][H]

    # ---------- 누적용 에러 카운트 ----------
    @th.inference_mode()
    def get_error_counts(
        self,
        src_wav: th.Tensor,   # reference
        out_wav: th.Tensor,   # hypothesis
        key = 'Output',
        remove_punct: Optional[bool] = None,  # None이면 WER=True / PhER=False 기본
        return_texts: bool = False,
    ):
        """
        단일 (src, out) 쌍에 대해 (err, ref_units)를 반환.
        - 분모 0인 경우 (0, 0) 반환 (외부 누적 계산이 NaN 안 되도록)
        """
        ref_text = self(src_wav)
        hyp_text = self(out_wav)
        if key != 'Output':
            print(f"clean: \n{ref_text}")
        print(f"{key}: \n{hyp_text}")

        ref_n = self._normalize_text(ref_text, remove_punct=remove_punct)
        hyp_n = self._normalize_text(hyp_text, remove_punct=remove_punct)

        # 토큰화
        ref_tok = ref_n.split()
        hyp_tok = hyp_n.split()

        # ----- PhER일 때만: 침묵 제거/중복 축소/폰셋 매핑 -----
        if self.metric_type == "pher":
            ref_tok = self._postprocess_phonemes(ref_tok)
            hyp_tok = self._postprocess_phonemes(hyp_tok)

        if len(ref_tok) == 0:
            result = (0, 0)
            if return_texts:
                return (*result, ref_text, hyp_text)
            return result

        err = self._edit_distance(ref_tok, hyp_tok)
        result = (err, len(ref_tok))
        if return_texts:
            return (*result, ref_text, hyp_text)
        return result

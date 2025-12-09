# vi_spiritlm/model/utils.py
import torch
import torchaudio
from typing import List, Optional, Tuple, Iterable
from ..constants import SPEECH_START, SPEECH_END, TEXT_START, TEXT_END, HUBERT_PREFIX, DEFAULT_SR

def convert_to_wav_tensor(
    src, target_sr: int = DEFAULT_SR, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    src: đường dẫn file .wav/.flac hoặc (waveform_tensor, sr)
    return: mono waveform (1, T) float32 in [-1, 1] trên device (nếu chỉ định)
    """
    if isinstance(src, tuple):
        wav, sr = src
    elif isinstance(src, torch.Tensor):
        # assume mono & known sr (target_sr)
        wav, sr = src, target_sr
    else:
        # path-like string
        wav, sr = torchaudio.load(str(src))

    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)  # stereo -> mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    wav = wav.to(torch.float32)
    if device is not None:
        wav = wav.to(device)
    return wav  # (1, T)

def _ids_startswith(tokenizer, prefix: str) -> List[int]:
    return [tid for tok, tid in tokenizer.get_vocab().items() if tok.startswith(prefix)]

def get_bad_words_for_text_mode(tokenizer) -> List[List[int]]:
    """
    Cấm toàn bộ token thuộc speech khi generate TEXT-ONLY.
    Gồm: [SpeechStart], [SpeechEnd], và toàn bộ [HUBERT_*]
    (Bạn có thể bổ sung Pitch/Style nếu tokenizer có).
    """
    ids = []
    hubert_ids = _ids_startswith(tokenizer, HUBERT_PREFIX)
    ids.extend(hubert_ids)
    for t in (SPEECH_START, SPEECH_END):
        if t in tokenizer.get_vocab():
            ids.append(tokenizer.convert_tokens_to_ids(t))
    return [[i] for i in ids]

def get_bad_words_for_speech_mode(tokenizer) -> List[List[int]]:
    """
    Cấm các marker text khi generate SPEECH-ONLY.
    (tuỳ chiến lược, có thể cấm BPE text hoàn toàn; ở đây chỉ cấm markers)
    """
    ids = []
    for t in (TEXT_START, TEXT_END):
        if t in tokenizer.get_vocab():
            ids.append(tokenizer.convert_tokens_to_ids(t))
    return [[i] for i in ids]

def does_prompt_end_with_speech(tokenizer, input_ids: torch.Tensor) -> bool:
    """
    Kiểm tra xem prompt kết thúc bằng [SpeechStart] (mở block speech chưa đóng)
    """
    ss_id = tokenizer.convert_tokens_to_ids(SPEECH_START) if SPEECH_START in tokenizer.get_vocab() else None
    se_id = tokenizer.convert_tokens_to_ids(SPEECH_END) if SPEECH_END in tokenizer.get_vocab() else None
    if ss_id is None or se_id is None:
        return False
    ids = input_ids.tolist()
    last_ss = max((i for i, x in enumerate(ids) if x == ss_id), default=-1)
    last_se = max((i for i, x in enumerate(ids) if x == se_id), default=-1)
    return last_ss > last_se

def build_interleaved_prompt(chunks: Iterable[Tuple[str, str]]) -> str:
    """
    chunks: list of ("text" | "speech", payload_as_string)
      - Với "speech": payload phải là chuỗi token HuBERT rời rạc như "[HUBERT_2][HUBERT_2]...[HUBERT_5]"
      - Với "text"  : payload là văn bản
    Tự chèn [TextStart]/[TextEnd] & [SpeechStart]/[SpeechEnd] cho mỗi block.
    """
    blocks = []
    for kind, payload in chunks:
        if kind == "text":
            blocks.append(f"{TEXT_START}{payload}{TEXT_END}")
        elif kind == "speech":
            blocks.append(f"{SPEECH_START}{payload}{SPEECH_END}")
        else:
            raise ValueError(f"Unknown chunk kind: {kind}")
    return "".join(blocks)

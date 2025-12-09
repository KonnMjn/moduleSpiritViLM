# vi_spiritlm/model/model.py
import os
import torch
from typing import Dict, Any, Optional, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from ..constants import SPEECH_START, SPEECH_END, TEXT_START, TEXT_END, HUBERT_PREFIX
from .utils import (
    get_bad_words_for_text_mode, get_bad_words_for_speech_mode,
    does_prompt_end_with_speech, build_interleaved_prompt
)
from ..speech_tokenizer import load_speech_side

class ViSpiritLM:
    """
    Wrapper chạy inference kiểu SpiritLM-Base:
    - Nạp Vistral + LoRA adapters + tokenizer (đã có [HUBERT_*], [SpeechStart]/[End], [TextStart]/[End])
    - Build prompt đan xen
    - Generate TEXT hoặc SPEECH (trả về chuỗi units) với bad_words_ids phù hợp
    - Encode audio -> units (khi cần đưa audio vào prompt) thông qua speech side
    """
    def __init__(
        self,
        base_model_id_or_path: str,
        lora_path: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Load tokenizer từ LoRA dir (đảm bảo chứa special tokens và dải [HUBERT_*])
        self.tokenizer = AutoTokenizer.from_pretrained(lora_path, use_fast=True)
        # pad_token xử lý an toàn: dùng eos làm pad nếu chưa có pad token
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load LM base + gắn LoRA
        # kwargs = {}
        # if torch_dtype is not None:
        #     kwargs["torch_dtype"] = torch_dtype
        # self.model = AutoModelForCausalLM.from_pretrained(base_model_id_or_path, **kwargs)
        # self.model = PeftModel.from_pretrained(self.model, lora_path)
        # self.model.to(self.device)
        # self.model.eval()
        
        # 1) Load base
        kwargs = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(base_model_id_or_path, **kwargs)

        # 2) Resize base embeddings để khớp vocab của adapter
        expected_vocab = len(self.tokenizer)
        current_vocab  = self.model.get_input_embeddings().num_embeddings
        if current_vocab != expected_vocab:
            self.model.resize_token_embeddings(expected_vocab)

        # 3) Buộc ràng buộc lại (tie) giữa embed_tokens và lm_head
        #    (đặc biệt quan trọng vì cảnh báo tie_word_embeddings bạn thấy)
        self.model.config.tie_word_embeddings = True
        try:
            self.model.tie_weights()
        except Exception:
            pass

        # 4) Gắn LoRA
        self.model = PeftModel.from_pretrained(self.model, lora_path)  # is_trainable=False mặc định
        self.model.to(self.device).eval()

        # Speech side (HuBERT + KMeans + (stub) HiFiGAN)
        self.speech = load_speech_side(mode="base", device=self.device)

    # ---------- Prompt helpers ----------
    def encode_prompt(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def build_prompt(self, chunks: List[Tuple[str, str]]) -> str:
        return build_interleaved_prompt(chunks)

    # ---------- Inference ----------
    @torch.inference_mode()
    def _generate(
        self,
        input_ids: torch.Tensor,
        bad_words_ids: Optional[List[List[int]]] = None,
        **gen_kwargs: Any,
    ) -> torch.Tensor:
        if "pad_token_id" not in gen_kwargs or gen_kwargs["pad_token_id"] is None:
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        return self.model.generate(
            input_ids=input_ids,
            bad_words_ids=bad_words_ids,
            **gen_kwargs
        )

    @torch.inference_mode()
    def generate_text(
        self,
        prompt: str,
        **gen_kwargs: Any,
    ) -> str:
        """
        Sinh TEXT-ONLY. Nếu prompt kết thúc bằng [SpeechStart] chưa đóng, tự chèn [SpeechEnd].
        """
        if SPEECH_START in prompt and (SPEECH_END not in prompt or prompt.rfind(SPEECH_START) > prompt.rfind(SPEECH_END)):
            prompt = prompt + SPEECH_END
        if TEXT_START not in prompt or prompt.rfind(TEXT_START) < prompt.rfind(TEXT_END) if TEXT_END in prompt else True:
            # đảm bảo khối text ở cuối để model hiểu cần sinh text
            prompt = prompt + TEXT_START

        input_ids = self.encode_prompt(prompt)
        bad = get_bad_words_for_text_mode(self.tokenizer)
        out = self._generate(input_ids, bad_words_ids=bad, **gen_kwargs)
        decoded = self.tokenizer.decode(out[0], skip_special_tokens=False)

        # Lấy phần text sau khối TEXT_START cuối cùng đến TEXT_END hoặc tới hết
        start = decoded.rfind(TEXT_START)
        if start >= 0:
            start += len(TEXT_START)
            end = decoded.find(TEXT_END, start)
            return decoded[start:end] if end >= 0 else decoded[start:]
        return decoded

    @torch.inference_mode()
    def generate_speech_units(
        self,
        prompt: str,
        **gen_kwargs: Any,
    ) -> List[int]:
        """
        Sinh SPEECH-ONLY (trả về danh sách unit id HuBERT). 
        Chưa tổng hợp waveform vì bạn chưa tích hợp HiFiGAN.
        """
        if TEXT_START in prompt and (TEXT_END not in prompt or prompt.rfind(TEXT_START) > prompt.rfind(TEXT_END)):
            prompt = prompt + TEXT_END
        if SPEECH_START not in prompt or prompt.rfind(SPEECH_START) < prompt.rfind(SPEECH_END) if SPEECH_END in prompt else True:
            prompt = prompt + SPEECH_START

        input_ids = self.encode_prompt(prompt)
        bad = get_bad_words_for_speech_mode(self.tokenizer)
        out = self._generate(input_ids, bad_words_ids=bad, **gen_kwargs)
        decoded = self.tokenizer.decode(out[0], skip_special_tokens=False)

        # Lấy chuỗi units giữa [SpeechStart] ... [SpeechEnd]
        start = decoded.rfind(SPEECH_START)
        if start < 0:
            return []
        start += len(SPEECH_START)
        end = decoded.find(SPEECH_END, start)
        unit_str = decoded[start:end] if end >= 0 else decoded[start:]
        return self.speech.string_to_units(unit_str)

    # ---------- Utilities for audio prompts ----------
    @torch.inference_mode()
    def encode_audio_to_units(self, wav_tensor: torch.Tensor) -> List[int]:
        """
        wav_tensor: (1, T) float32 @ 16kHz
        """
        return self.speech.encode_audio_to_units(wav_tensor)

    def units_to_string(self, units: List[int]) -> str:
        return self.speech.units_to_string(units)

    def string_to_units(self, s: str) -> List[int]:
        return self.speech.string_to_units(s)

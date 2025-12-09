# vi_spiritlm/speech_tokenizer/tokenizer.py
from typing import List
from ..constants import HUBERT_PREFIX

class UnitsStringCodec:
    """
    Chuyển giữa units (0..99) <-> chuỗi rời rạc "[HUBERT_i]".
    Kèm dedup consecutive cho phù hợp SpiritLM-Base.
    """
    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size

    def units_to_string(self, units: List[int]) -> str:
        toks = []
        for u in units:
            if not (0 <= u < self.vocab_size):
                raise ValueError(f"Unit id {u} out of range [0, {self.vocab_size-1}]")
            toks.append(f"{HUBERT_PREFIX}{u}]")
        return "".join(toks)

    def string_to_units(self, s: str) -> List[int]:
        units = []
        i = 0
        prefix = HUBERT_PREFIX
        plen = len(prefix)
        n = len(s)
        while i < n:
            j = s.find(prefix, i)
            if j < 0:
                break
            k = s.find("]", j + plen)
            if k < 0:
                break
            num_str = s[j + plen : k]
            try:
                u = int(num_str)
                if 0 <= u < self.vocab_size:
                    units.append(u)
            except Exception:
                pass
            i = k + 1
        return units

    @staticmethod
    def dedup(units: List[int]) -> List[int]:
        if not units:
            return units
        out = [units[0]]
        for u in units[1:]:
            if u != out[-1]:
                out.append(u)
        return out

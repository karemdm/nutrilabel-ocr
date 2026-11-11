"""Adapters for vision-language OCR models (PaddleOCR-VL, olmOCR, GLM-OCR, Chandra...).

These models return markdown / HTML instead of word boxes. Their output is laid out
on a synthetic grid by :func:`nutrilabel.extraction.markup.text_to_words`, so the same
extractor and benchmark apply - but the boxes are *not* image coordinates.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from nutrilabel.engines.base import EngineNotAvailable, OCREngine
from nutrilabel.types import Word

VLM_PROMPT = (
    "Transcribe the nutrition facts table in this image as a markdown table. "
    "Copy every row label and every number with its unit exactly as printed, keep the "
    "column headers (e.g. 'per 100 g', 'per serving', '% RI'). Do not convert, compute "
    "or guess values. If there is no nutrition table, answer 'NONE'."
)


class PaddleOCRVLEngine(OCREngine):
    """PaddleOCR-VL (document-parsing VLM, ``paddleocr>=3.2``). Experimental; a GPU is
    strongly recommended."""

    name: ClassVar[str] = "paddleocr-vl"
    pip_extra: ClassVar[str] = "paddle-vl"
    boxes_are_synthetic: ClassVar[bool] = True

    def __init__(self, languages=None, **pipeline_kwargs: Any) -> None:
        super().__init__(languages)
        self.pipeline_kwargs = pipeline_kwargs
        self._pipeline = None

    @classmethod
    def _check_import(cls) -> None:
        from paddleocr import PaddleOCRVL  # noqa: F401

    def _run(self, image: np.ndarray) -> list[Word]:
        from nutrilabel.extraction.markup import text_to_words

        if self._pipeline is None:
            from paddleocr import PaddleOCRVL

            self._pipeline = PaddleOCRVL(**self.pipeline_kwargs)
        output = self._pipeline.predict(image[:, :, ::-1].copy())
        return text_to_words(paddle_vl_to_text(output))


def paddle_vl_to_text(output: Any) -> str:
    """Collect the markdown (or block contents) of PaddleOCR-VL results."""
    parts: list[str] = []
    for res in output or []:
        md = getattr(res, "markdown", None)
        if isinstance(md, dict) and md.get("markdown_texts"):
            parts.append(str(md["markdown_texts"]))
            continue
        get = res.get if hasattr(res, "get") else lambda k, d=None, r=res: getattr(r, k, d)
        for block in get("parsing_res_list", []) or []:
            content = block.get("block_content") if isinstance(block, dict) else \
                getattr(block, "content", None) or getattr(block, "block_content", None)
            if content:
                parts.append(str(content))
    return "\n".join(parts)


class VLMEngine(OCREngine):
    """Any vision-language model behind an OpenAI-compatible endpoint, e.g. a local
    Ollama / vLLM / LM Studio server running GLM-OCR, olmOCR, Qwen2.5-VL, MiniCPM-V...

    Configure with arguments or environment variables::

        NUTRILABEL_VLM_URL=http://localhost:11434/v1   # Ollama default
        NUTRILABEL_VLM_MODEL=qwen2.5vl:7b
        NUTRILABEL_VLM_API_KEY=...                     # only if your server needs one

    The engine is "available" only when a model name is configured, so ``engine="auto"``
    never calls a server by surprise. Keep the server local: product photos are sent to it.
    """

    name: ClassVar[str] = "vlm"
    pip_extra: ClassVar[str] = ""
    boxes_are_synthetic: ClassVar[bool] = True

    def __init__(self, languages=None, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, prompt: str = VLM_PROMPT, timeout: float = 180.0,
                 max_side: int = 1280) -> None:
        import os

        super().__init__(languages)
        self.model = model or os.environ.get("NUTRILABEL_VLM_MODEL", "")
        self.base_url = (base_url or os.environ.get("NUTRILABEL_VLM_URL",
                                                    "http://localhost:11434/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("NUTRILABEL_VLM_API_KEY", "")
        self.prompt = prompt
        self.timeout = timeout
        self.max_side = max_side

    @classmethod
    def _check_import(cls) -> None:
        import os

        if not os.environ.get("NUTRILABEL_VLM_MODEL"):
            raise EngineNotAvailable("set NUTRILABEL_VLM_MODEL (and NUTRILABEL_VLM_URL)")

    def request_payload(self, image: np.ndarray) -> dict[str, Any]:
        import base64
        import io

        from PIL import Image

        img = Image.fromarray(image)
        img.thumbnail((self.max_side, self.max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return {
            "model": self.model,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
        }

    def _run(self, image: np.ndarray) -> list[Word]:
        import json
        import urllib.request

        from nutrilabel.extraction.markup import text_to_words

        if not self.model:
            raise EngineNotAvailable("no VLM model configured (NUTRILABEL_VLM_MODEL)")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(self.request_payload(image)).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"] or ""
        if text.strip().upper() == "NONE":
            return []
        return text_to_words(text)


class HFVLMEngine(OCREngine):
    """Open-weight OCR / vision-language model run locally with Hugging Face ``transformers``.

    Subclasses only set ``model_id`` (see :data:`HF_VLM_PRESETS`). Any other
    image-text-to-text checkpoint can be used with ``HFVLMEngine(model_id=...)``.
    A CUDA GPU is strongly recommended (7B models take minutes per image on CPU).
    """

    name: ClassVar[str] = "hf-vlm"
    pip_extra: ClassVar[str] = "vlm"
    boxes_are_synthetic: ClassVar[bool] = True
    model_id: ClassVar[str] = "Qwen/Qwen2.5-VL-3B-Instruct"
    trust_remote_code: ClassVar[bool] = False

    def __init__(self, languages=None, model_id: str | None = None, prompt: str = VLM_PROMPT,
                 max_new_tokens: int = 1024, max_side: int = 1280, device_map: str = "auto",
                 torch_dtype: str = "auto") -> None:
        super().__init__(languages)
        self.model_id_ = model_id or self.model_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.max_side = max_side
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self._model = None
        self._processor = None

    @classmethod
    def _check_import(cls) -> None:
        import torch  # noqa: F401
        import transformers  # noqa: F401

    def _load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        kw = {"trust_remote_code": self.trust_remote_code}
        self._processor = AutoProcessor.from_pretrained(self.model_id_, **kw)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id_, device_map=self.device_map, **self._dtype_kwarg(), **kw
        ).eval()

    def _dtype_kwarg(self) -> dict[str, Any]:
        """``dtype=`` on transformers >= 4.56, ``torch_dtype=`` before."""
        import transformers

        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        return {"dtype" if (major, minor) >= (4, 56) else "torch_dtype": self.torch_dtype}

    def generate_text(self, image: np.ndarray) -> str:
        import torch
        from PIL import Image

        if self._model is None:
            self._load()
        img = Image.fromarray(image)
        img.thumbnail((self.max_side, self.max_side))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": self.prompt},
        ]}]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                       do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    def _run(self, image: np.ndarray) -> list[Word]:
        from nutrilabel.extraction.markup import text_to_words

        text = self.generate_text(image)
        if text.strip().upper() == "NONE":
            return []
        return text_to_words(text)


#: Free, open-weight OCR models (engine name -> Hugging Face checkpoint).
HF_VLM_PRESETS: dict[str, tuple[str, bool]] = {
    # name: (model id, needs trust_remote_code)
    "olmocr": ("allenai/olmOCR-2-7B-1025", False),  # Apache-2.0, Qwen2.5-VL fine-tune
    "glm-ocr": ("zai-org/GLM-OCR", False),  # needs a recent transformers release
    "chandra": ("datalab-to/chandra-ocr-2", False),  # OpenRAIL (non-commercial use is fine)
    "nanonets-ocr": ("nanonets/Nanonets-OCR2-3B", False),
    "deepseek-ocr": ("deepseek-ai/DeepSeek-OCR-2", True),  # custom code; vLLM recommended
    "qwen-vl": ("Qwen/Qwen2.5-VL-3B-Instruct", False),  # general VLM baseline
}


def _make_preset(engine_name: str, model_id: str, remote: bool) -> type[HFVLMEngine]:
    cls_name = "".join(p.capitalize() for p in engine_name.split("-")) + "Engine"
    return type(cls_name, (HFVLMEngine,), {
        "name": engine_name, "model_id": model_id, "trust_remote_code": remote,
        "__doc__": f"``{model_id}`` via transformers (see :class:`HFVLMEngine`).",
    })


class DeepSeekOCREngine(HFVLMEngine):
    """DeepSeek-OCR(-2) ships its own inference code (``model.infer``) instead of the
    standard chat template, following its model card. Experimental: for throughput,
    serve it with vLLM and use the ``vlm`` engine instead."""

    name: ClassVar[str] = "deepseek-ocr"
    model_id: ClassVar[str] = "deepseek-ai/DeepSeek-OCR-2"
    trust_remote_code: ClassVar[bool] = True

    @classmethod
    def _check_import(cls) -> None:
        import torch

        super()._check_import()
        if not torch.cuda.is_available():  # its remote code calls .cuda() unconditionally
            raise EngineNotAvailable("DeepSeek-OCR needs an NVIDIA GPU (CUDA)")

    def _load(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        self._processor = AutoTokenizer.from_pretrained(self.model_id_, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            self.model_id_, trust_remote_code=True, use_safetensors=True,
            device_map=self.device_map, **self._dtype_kwarg(),
        ).eval()

    def generate_text(self, image: np.ndarray) -> str:
        import tempfile
        from pathlib import Path

        from PIL import Image

        if self._model is None:
            self._load()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "label.jpg"
            Image.fromarray(image).save(path, quality=95)
            out = self._model.infer(
                self._processor, prompt="<image>\n" + self.prompt, image_file=str(path),
                output_path=tmp, base_size=1024, image_size=640, crop_mode=True,
                save_results=False, eval_mode=True,
            )
        return out if isinstance(out, str) else str(out or "")


HF_VLM_ENGINES: list[type[HFVLMEngine]] = [
    DeepSeekOCREngine if n == "deepseek-ocr" else _make_preset(n, mid, remote)
    for n, (mid, remote) in HF_VLM_PRESETS.items()
]

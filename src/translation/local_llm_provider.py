"""
local_llm_provider.py
Proveedor de traducción contextual mediante LLMs locales (Ollama / LM Studio).
100% offline y local (localhost / 127.0.0.1).
Permite aprovechar modelos locales (como Llama 3, Qwen 2.5, Gemma 2) para traducciones
con comprensión de contexto multi-turno, modismos y jerga sin enviar ningún dato a Internet.
"""

import json
import urllib.request
import urllib.error
from typing import Optional, List, Tuple
import logging

from src.translation.base import TranslationProvider
from src.translation.glossary import SmartGlossary

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"


class LocalLLMTranslationProvider(TranslationProvider):
    """
    Proveedor de traducción vía API local de Ollama o LM Studio (OpenAI-compatible).
    Garantía 100% offline: rechaza cualquier URL que no sea localhost o 127.0.0.1.
    """

    def __init__(
        self,
        endpoint_url: str = DEFAULT_OLLAMA_URL,
        model_name: str = "llama3.2",
        timeout_seconds: float = 2.0,
        glossary: Optional[SmartGlossary] = None,
    ):
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.glossary = glossary or SmartGlossary()
        self._validate_local_only()

    def _validate_local_only(self) -> None:
        """Verifica estrictamente que la conexión apunte a la máquina local."""
        if not (
            self.endpoint_url.startswith("http://127.0.0.1")
            or self.endpoint_url.startswith("http://localhost")
        ):
            raise ValueError(
                f"Modo offline estricto: El endpoint '{self.endpoint_url}' no es local (127.0.0.1 o localhost)."
            )

    def is_available(self) -> bool:
        """Comprueba rápidamente (en <300ms) si el servidor local está respondiendo."""
        try:
            base_url = "http://127.0.0.1:11434/" if "11434" in self.endpoint_url else "http://127.0.0.1:1234/v1/models"
            req = urllib.request.Request(base_url, headers={"User-Agent": "TraductorTiempoReal/1.0"})
            with urllib.request.urlopen(req, timeout=0.3) as response:
                return response.status in (200, 404)
        except Exception:
            return False

    def translate(
        self,
        text: str,
        source: str,
        target: str,
        context: Optional[List[str]] = None,
    ) -> str:
        clean = text.strip()
        if not clean or source == target:
            return clean

        prompt = self._build_prompt(clean, source, target, context)

        try:
            if "11434" in self.endpoint_url:
                # Formato Ollama /api/generate
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 100},
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    self.endpoint_url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    raw_result = resp_data.get("response", "").strip()

            else:
                # Formato LM Studio / OpenAI-compatible
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": f"You are an expert real-time translator from {source} to {target}. Translate accurately and concisely. Return ONLY the translated sentence, nothing else.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 100,
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    self.endpoint_url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    raw_result = resp_data["choices"][0]["message"]["content"].strip()

            # Limpiar comillas o prefijos que el LLM pudiera generar
            cleaned = self._clean_llm_output(raw_result)
            return self.glossary.apply(cleaned)

        except Exception as e:
            logger.warning("Fallo al traducir con LLM local (%s): %s", self.endpoint_url, e)
            raise RuntimeError(f"Servidor LLM local no disponible: {e}") from e

    def _build_prompt(
        self,
        text: str,
        source: str,
        target: str,
        context: Optional[List[str]],
    ) -> str:
        ctx_str = ""
        if context:
            recent = context[-3:]
            ctx_str = f"Previous conversation context:\n" + "\n".join(f"- {c}" for c in recent) + "\n\n"

        return (
            f"Translate the following speech sentence from {source} to {target}.\n"
            f"{ctx_str}"
            f"Text to translate: \"{text}\"\n"
            f"Translated text:"
        )

    def _clean_llm_output(self, output: str) -> str:
        res = output.strip()
        if (res.startswith('"') and res.endswith('"')) or (res.startswith("'") and res.endswith("'")):
            res = res[1:-1].strip()
        if res.lower().startswith("traducción:"):
            res = res[11:].strip()
        return res

    @property
    def provider_name(self) -> str:
        return "local_llm"

    @property
    def supported_pairs(self) -> List[Tuple[str, str]]:
        # Los LLMs modernos soportan cualquier par lingüístico común
        return [
            ("en", "es"),
            ("es", "en"),
            ("en", "pt"),
            ("pt", "en"),
            ("es", "pt"),
            ("pt", "es"),
        ]

    @property
    def is_ready(self) -> bool:
        return self.is_available()

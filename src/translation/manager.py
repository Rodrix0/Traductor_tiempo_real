"""
manager.py
Gestor central de traducción neuronal (TranslationManager).
Coordina la selección de proveedores modulares (Marian, Argos, LLM local),
mantiene la memoria de contexto conversacional y aplica el glosario inteligente.
"""

from typing import Optional, Dict, List, Tuple, Any
import logging

from src.translation.base import TranslationProvider
from src.translation.glossary import SmartGlossary
from src.translation.context_memory import ContextMemory
from src.translation.marian_provider import MarianTranslationProvider
from src.translation.argos_provider import ArgosTranslationProvider
from src.translation.local_llm_provider import LocalLLMTranslationProvider

logger = logging.getLogger(__name__)


class TranslationManager:
    """Administrador modular de traducción con enrutamiento inteligente y memoria multi-turno."""

    def __init__(
        self,
        preferred_provider: str = "auto",
        device: str = "cpu",
        context_turns: int = 5,
        enable_glossary: bool = True,
    ):
        self.preferred_provider = preferred_provider.lower()
        self.device = device
        self.glossary = SmartGlossary() if enable_glossary else None
        self.context_memory = ContextMemory(max_turns=context_turns)

        # Registro de proveedores instanciados
        self.marian = MarianTranslationProvider(device=device, glossary=self.glossary)
        self.argos = ArgosTranslationProvider(glossary=self.glossary)
        self.local_llm = LocalLLMTranslationProvider(glossary=self.glossary)

    def select_provider(self, source: str, target: str) -> TranslationProvider:
        """
        Selecciona el mejor proveedor disponible para el par lingüístico solicitado.
        Prioridad para en->es: MarianMT (ultra-rápido, alta fidelidad).
        Prioridad para otros pares: ArgosTranslate.
        Si se prefiere local_llm y está activo, se utiliza para cualquier par compatible.
        """
        pair = (source, target)

        if self.preferred_provider == "local_llm" and self.local_llm.is_ready:
            return self.local_llm

        if self.preferred_provider == "marian" and pair in self.marian.supported_pairs and self.marian.is_ready:
            return self.marian

        if self.preferred_provider == "argos" and pair in self.argos.supported_pairs and self.argos.is_ready:
            return self.argos

        # Selección automática inteligente
        if source == "en" and target == "es" and self.marian.is_ready:
            return self.marian

        if pair in self.argos.supported_pairs:
            return self.argos

        # Si Argos no lo tiene pero Marian sí
        if pair in self.marian.supported_pairs and self.marian.is_ready:
            return self.marian

        return self.argos

    def translate(
        self,
        text: str,
        source: str,
        target: str,
        use_context: bool = True,
    ) -> str:
        """
        Traduce un texto desde 'source' a 'target', consultando el contexto previo
        y actualizando la memoria de contexto tras la traducción exitosa.
        """
        clean = text.strip()
        if not clean or source == target:
            return clean

        provider = self.select_provider(source, target)
        context = self.context_memory.get_recent_source_context() if use_context else None

        try:
            translated = provider.translate(clean, source, target, context=context)
        except Exception as e:
            logger.warning("Fallo en proveedor %s: %s. Intentando proveedor de respaldo...", provider.provider_name, e)
            fallback = self.argos if provider != self.argos else self.marian
            try:
                translated = fallback.translate(clean, source, target, context=context)
            except Exception as fe:
                logger.error("Todos los proveedores fallaron para (%s -> %s): %s", source, target, fe)
                return clean

        # Registrar el turno en la memoria de contexto
        self.context_memory.add_turn(
            source_text=clean,
            target_text=translated,
            source_lang=source,
            target_lang=target,
        )

        return translated

    def clean_source(self, text: str, source: str) -> str:
        """Normaliza el texto de origen."""
        if source == "en":
            return self.marian.clean_source(text, source)
        return " ".join(text.split()).strip()

    def get_whisper_prompt(self) -> Optional[str]:
        """Obtiene vocabulario para Whisper desde el glosario activo."""
        if self.glossary:
            return self.marian.get_whisper_prompt()
        return None

    def reset_context(self) -> None:
        """Limpia la memoria de contexto (por ejemplo, al iniciar una nueva sesión)."""
        self.context_memory.clear()

    @property
    def is_marian_available(self) -> bool:
        return self.marian.is_ready

    @property
    def is_argos_available(self) -> bool:
        return self.argos.is_ready

    @property
    def is_local_llm_available(self) -> bool:
        return self.local_llm.is_ready

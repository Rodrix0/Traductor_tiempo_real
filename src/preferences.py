"""Validated preferences, written atomically."""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass
class Preferences:
    source: str = 'auto'
    target: str = 'es'
    model: str = 'base'
    compute: str = 'auto'
    device_name: str = ''
    font_size: int = 24
    opacity: float = 0.95
    threshold: float = 0.008
    chunk_seconds: float = 4.0

    def validate(self):
        for value, allowed in [(self.source, ('auto','es','en','pt')), (self.target, ('es','en','pt')),
                (self.model, ('base','small')), (self.compute, ('auto','cpu'))]:
            if value not in allowed:
                raise ValueError('Idioma, modelo o procesamiento no admitido.')
        if not isinstance(self.device_name, str):
            raise ValueError('Dispositivo inválido.')
        if type(self.font_size) is not int or not 16 <= self.font_size <= 48:
            raise ValueError('El tamaño de subtítulos debe estar entre 16 y 48.')
        for value, low, high in [(self.opacity,0.45,1), (self.threshold,0.002,0.05), (self.chunk_seconds,2,8)]:
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not low <= value <= high:
                raise ValueError('Opacidad, sensibilidad o duración fuera del rango permitido.')
        return self


def load_preferences(path):
    path = Path(path)
    if not path.exists():
        return Preferences(), ''
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('Se esperaba un objeto de configuración.')
        return Preferences(**{key: value for key,value in data.items() if key in {f.name for f in fields(Preferences)}}).validate(), ''
    except (OSError, ValueError, TypeError) as exc:
        return Preferences(), f'No se pudieron leer las preferencias; se usan valores iniciales: {exc}'


def save_preferences(path, preferences):
    preferences.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(asdict(preferences), ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

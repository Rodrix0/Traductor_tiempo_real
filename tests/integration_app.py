"""Opt-in full desktop worker integration with real models and temporary history."""
import tempfile
import time
import tkinter as tk
from pathlib import Path
from unittest.mock import patch
from traductor import TranslatorApp
from src.storage import HistoryStore
from src.preferences import load_preferences


def main():
    with tempfile.TemporaryDirectory() as folder:
        root = tk.Tk()
        root.withdraw()
        app = TranslatorApp(root, data_dir=folder)
        errors = []
        try:
            app.pref_origin.set('English')
            app.pref_target.set('Português')
            app.pref_compute.set('CPU')
            assert app.save_settings_action()
            assert load_preferences(Path(folder)/'preferences.json')[0].target == 'pt'
            app.file_path.set(str(Path('models/speech_smoke.wav').resolve()))
            with patch('traductor.messagebox.showerror', side_effect=lambda *args: errors.append(args)):
                app.launch_file()
                deadline = time.monotonic() + 180
                while app.worker.is_alive() and time.monotonic() < deadline:
                    root.update()
                    time.sleep(0.02)
                assert not app.worker.is_alive(), 'El trabajo no terminó en 180 segundos'
                app.poll()
                assert not errors, errors
                assert app.file_captions, 'La interfaz no recibió subtítulos'
                app.reload_history()
                row = app.store.search()[0]
                assert row['state'] == 'completado', row
                app.sessions.selection_set(row['id'])
                app.select_history()
                assert app.file_captions[0].translated in app.history_preview.get('1.0','end')
                for format in ('srt','vtt','txt','json'):
                    path = Path(folder)/f'export.{format}'
                    with patch('traductor.filedialog.asksaveasfilename', return_value=str(path)):
                        app.export_history(format)
                    assert path.is_file() and path.stat().st_size > 0
                reopened = HistoryStore(Path(folder)/'history.sqlite3')
                assert reopened.captions(row['id']) == app.file_captions
                assert app.engine.actual_device == 'cpu'
                print('ARCHIVO → INTERFAZ → HISTORIAL → 4 EXPORTACIONES → REAPERTURA: OK', flush=True)
                print('Preferencias persistentes EN → PT / CPU: OK', flush=True)
        finally:
            app.stopped.set()
            while app.worker and app.worker.is_alive():
                root.update()
                time.sleep(0.02)
            app.close()


if __name__ == '__main__':
    main()

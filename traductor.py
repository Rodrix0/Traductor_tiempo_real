"""Desktop entry point for automatic translated subtitles on Windows."""
import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from src.subtitles import Caption, export_captions
from src.storage import HistoryStore
from src.preferences import load_preferences
from src.ui.settings_panel import SettingsPanel, configure_theme, style_text_widgets
from src.translation.local import LANGUAGES, LocalTranslator, prepare_models
from src.subtitles_dialogue import DialogueManager

logger = logging.getLogger(__name__)


class TranslatorApp(SettingsPanel):
    def __init__(self, root, data_dir=None):
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        self.root = root
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent / 'data'
        self.store = HistoryStore(self.data_dir / 'history.sqlite3')
        self.preferences, preferences_warning = load_preferences(self.data_dir / 'preferences.json')
        self.snapshot_processing()
        self.closing = False
        root.title("Traductor Studio · ES / EN / PT")
        root.geometry("1020x740")
        root.minsize(880, 620)
        configure_theme(root)
        self.events = queue.Queue(maxsize=100)
        self.stopped = threading.Event()
        self.worker = None
        self.capture = None
        self.pipeline_controller = None
        self.engine = None
        self.engine_size = None
        self.translator = LocalTranslator()
        self.overlay = None
        self.dialogue_manager = DialogueManager()
        self.caption_updated = 0
        self.file_captions = []
        self.file_path = tk.StringVar()
        self.original = tk.StringVar(value="Esperando audio…")
        self.subtitle = tk.StringVar(value="Los subtítulos traducidos aparecerán aquí.")
        self.status = tk.StringVar(value="Prepará los idiomas una vez y después presioná Iniciar.")
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True)
        body = ttk.Frame(self.tabs, padding=20)
        self.tabs.add(body, text="  En vivo  ")
        ttk.Label(body, text="Traductor en vivo", font=("Segoe UI", 23, "bold")).pack(anchor="w")
        ttk.Label(body, text="Videos, llamadas, películas y series · traducción local").pack(anchor="w", pady=(0, 15))
        ttk.Label(body, text="Fuente de audio (en llamadas, elegí los auriculares que usa la llamada)").pack(anchor="w")
        row = ttk.Frame(body)
        row.pack(fill="x", pady=(4, 12))
        self.source = ttk.Combobox(row, state="readonly", width=72)
        self.source.pack(side="left", fill="x", expand=True)
        self.refresh = ttk.Button(row, text="Actualizar", command=self.load_sources)
        self.refresh.pack(side="left", padx=(8, 0))
        row = ttk.Frame(body)
        row.pack(fill="x")
        self.origin = self.combo(row, "Idioma del audio", ["Automático"] + list(LANGUAGES.values()), 0)
        self.target = self.combo(row, "Traducir a", list(LANGUAGES.values()), 0)
        self.model = self.combo(row, "Reconocimiento", ["small.en", "small", "base"], 0)
        row = ttk.Frame(body)
        row.pack(fill="x", pady=15)
        self.prepare = ttk.Button(row, text="Preparar idiomas", command=lambda: self.launch(True))
        self.prepare.pack(side="left")
        self.start = ttk.Button(row, text="▶ Iniciar", command=lambda: self.launch(False))
        self.start.pack(side="left", padx=8)
        self.stop = ttk.Button(row, text="■ Detener", command=self.request_stop, state="disabled")
        self.stop.pack(side="left")
        ttk.Button(row, text="Ventana de subtítulos", command=self.show_overlay).pack(side="right")
        ttk.Label(body, textvariable=self.status, wraplength=840).pack(anchor="w")
        self.meter = ttk.Progressbar(body, maximum=0.12)
        self.meter.pack(fill="x", pady=(5, 10))
        ttk.Label(body, textvariable=self.original, wraplength=830, foreground="#aebbd0").pack(anchor="w")
        ttk.Label(body, textvariable=self.subtitle, wraplength=830, font=("Segoe UI", 19, "bold")).pack(anchor="w", pady=12)
        self.history = tk.Text(body, height=7, wrap="word", state="disabled", font=("Segoe UI", 11))
        self.history.pack(fill="both", expand=True)
        ttk.Label(body, text="Procesamiento local. Las transcripciones se guardan en Historial; no se guarda audio.\nCaptura todo lo que suena en el dispositivo elegido. Usá pantalla completa sin exclusividad para ver los subtítulos.").pack(anchor="w", pady=(10, 0))
        self.controls = [self.source, self.origin, self.target, self.model, self.refresh, self.prepare, self.start]
        self.build_files_tab()
        self.build_history_tab()
        self.build_settings_tab()
        self.load_sources()
        self.apply_preferences()
        if preferences_warning:
            self.status.set(preferences_warning)
        style_text_widgets(root)
        from src.ui.hotkeys import bind_local_hotkeys
        bind_local_hotkeys(
            root,
            on_toggle_visibility=self.toggle_overlay,
            on_toggle_clickthrough=lambda: self.overlay.toggle_click_through() if self.overlay and self.overlay.exists() else None,
            on_font_increase=lambda: self.overlay.change_font_size(2) if self.overlay and self.overlay.exists() else None,
            on_font_decrease=lambda: self.overlay.change_font_size(-2) if self.overlay and self.overlay.exists() else None,
        )
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    def build_files_tab(self):
        page = ttk.Frame(self.tabs, padding=20)
        self.tabs.add(page, text="  Archivos  ")
        ttk.Label(page, text="Subtítulos de audio y video", font=("Segoe UI", 22, "bold")).pack(anchor="w")
        ttk.Label(page, text="Importá un archivo y exportá la traducción con sus tiempos originales.").pack(anchor="w", pady=8)
        row = ttk.Frame(page)
        row.pack(fill="x", pady=10)
        ttk.Entry(row, textvariable=self.file_path, state="readonly").pack(side="left", fill="x", expand=True)
        self.pick_file = ttk.Button(row, text="Elegir archivo…", command=self.choose_file)
        self.pick_file.pack(side="left", padx=8)
        row = ttk.Frame(page)
        row.pack(fill="x", pady=8)
        self.file_origin = self.combo(row, "Idioma del audio", ['Automático'] + list(LANGUAGES.values()), 0)
        self.file_target = self.combo(row, "Traducir a", list(LANGUAGES.values()), 0)
        self.file_model = self.combo(row, "Reconocimiento", ['small.en', 'small', 'base'], 0)
        row = ttk.Frame(page)
        row.pack(fill="x", pady=12)
        self.process_file = ttk.Button(row, text="Generar subtítulos", command=self.launch_file)
        self.process_file.pack(side="left")
        self.cancel_file = ttk.Button(row, text="Cancelar", command=self.request_stop, state="disabled")
        self.cancel_file.pack(side="left", padx=8)
        for format in ('srt', 'vtt', 'txt'):
            ttk.Button(row, text=f'Exportar {format.upper()}', command=lambda f=format: self.export_file(f)).pack(side="right", padx=3)
        ttk.Label(page, textvariable=self.status, wraplength=820).pack(anchor="w", pady=8)
        self.file_preview = tk.Text(page, wrap='word', state='disabled', font=('Segoe UI', 12))
        self.file_preview.pack(fill='both', expand=True)
        ttk.Label(page, text='Se procesa la primera pista de audio. Cancelar conserva el resultado parcial para exportarlo.').pack(anchor='w', pady=8)
        self.controls.extend([self.pick_file, self.process_file, self.file_origin, self.file_target, self.file_model])

    def build_history_tab(self):
        page = ttk.Frame(self.tabs, padding=20)
        self.tabs.add(page, text='  Historial  ')
        ttk.Label(page, text='Tus transcripciones', font=('Segoe UI', 22, 'bold')).pack(anchor='w')
        ttk.Label(page, text='Sesiones guardadas en esta computadora · búsqueda por nombre o contenido · últimas 200 coincidencias').pack(anchor='w', pady=8)
        row = ttk.Frame(page)
        row.pack(fill='x', pady=8)
        self.search_text = tk.StringVar()
        search = ttk.Entry(row, textvariable=self.search_text)
        search.pack(side='left', fill='x', expand=True)
        search.bind('<Return>', lambda _: self.reload_history())
        ttk.Button(row, text='Buscar / Actualizar', command=self.reload_history).pack(side='left', padx=8)
        self.sessions = ttk.Treeview(page, columns=('name','mode','target','state','count'), show='headings', height=7)
        for key, title, width in [('name','Sesión / fecha',310),('mode','Origen',80),('target','Destino',75),('state','Estado',90),('count','Frases',65)]:
            self.sessions.heading(key, text=title)
            self.sessions.column(key, width=width, minwidth=50)
        self.sessions.pack(fill='x', pady=8)
        self.sessions.bind('<<TreeviewSelect>>', lambda _: self.select_history())
        row = ttk.Frame(page)
        row.pack(fill='x', pady=8)
        ttk.Label(row, text='Exportar sesión seleccionada:').pack(side='left')
        for format in ('txt', 'srt', 'vtt', 'json'):
            ttk.Button(row, text=format.upper(), command=lambda f=format: self.export_history(f)).pack(side='left', padx=4)
        self.history_preview = tk.Text(page, state='disabled', wrap='word', font=('Segoe UI', 11))
        self.history_preview.pack(fill='both', expand=True)
        self.reload_history()

    def reload_history(self):
        selected = self.sessions.selection()
        self.sessions.delete(*self.sessions.get_children())
        self.session_rows = {row['id']: row for row in self.store.search(self.search_text.get().strip())}
        for key, row in self.session_rows.items():
            from datetime import datetime
            date = datetime.fromisoformat(row['created']).astimezone().strftime('%d/%m %H:%M')
            self.sessions.insert('', 'end', iid=key, values=(f"{row['name']} · {date}", row['mode'], row['target'].upper(), row['state'], row['count']))
        if selected and selected[0] in self.session_rows:
            self.sessions.selection_set(selected[0])

    def select_history(self):
        selected = self.sessions.selection()
        self.history_preview.configure(state='normal')
        self.history_preview.delete('1.0','end')
        if selected:
            captions = self.store.captions(selected[0], limit=1000)
            self.history_preview.insert('end', export_captions(captions, 'txt'))
            if self.session_rows[selected[0]]['count'] > 1000:
                self.history_preview.insert('end', '\nVista previa: primeras 1000 frases. La exportación incluye la sesión completa.')
        self.history_preview.configure(state='disabled')

    def export_history(self, format):
        selected = self.sessions.selection()
        if not selected:
            messagebox.showinfo('Historial', 'Seleccioná una sesión para exportar.')
            return
        captions = self.store.captions(selected[0])
        if not captions:
            messagebox.showinfo('Historial', 'Esta sesión no tiene subtítulos guardados.')
            return
        self.save_export(captions, format, 'transcripcion')

    def choose_file(self):
        path = filedialog.askopenfilename(filetypes=[('Audio y video', '*.wav *.mp3 *.flac *.m4a *.ogg *.mp4 *.mkv *.mov *.webm *.avi'), ('Todos los archivos', '*.*')])
        if path:
            self.file_path.set(path)

    def export_file(self, format):
        if not self.file_captions:
            messagebox.showinfo('Sin subtítulos', 'Primero generá los subtítulos de un archivo.')
            return
        self.save_export(self.file_captions, format, Path(self.file_path.get()).stem or 'subtitulos')

    def save_export(self, captions, format, name):
        path = filedialog.asksaveasfilename(initialfile=f'{name}.{format}', defaultextension=f'.{format}',
            filetypes=[(format.upper(), f'*.{format}')])
        if path:
            try:
                Path(path).write_text(export_captions(captions, format), encoding='utf-8')
                self.status.set(f'Exportado: {path}')
            except OSError as exc:
                messagebox.showerror('No se pudo exportar', str(exc))

    def launch_file(self):
        if self.closing or (self.worker and self.worker.is_alive()):
            return
        if not Path(self.file_path.get()).is_file():
            messagebox.showerror('Archivo', 'Elegí un archivo de audio o video.')
            return
        reverse = {name: code for code, name in LANGUAGES.items()}
        source, target = reverse.get(self.file_origin.get()), reverse[self.file_target.get()]
        self.file_captions = []
        self.file_preview.configure(state='normal')
        self.file_preview.delete('1.0', 'end')
        self.file_preview.configure(state='disabled')
        self.stopped.clear()
        self.snapshot_processing()
        for control in self.controls:
            control.configure(state='disabled')
        self.cancel_file.configure(state='normal')
        self.worker = threading.Thread(target=self.run_file, args=(self.file_path.get(), source, target, self.file_model.get()), daemon=True)
        self.worker.start()

    def run_file(self, path, source, target, model):
        session = None
        state = 'completado'
        try:
            session = self.store.create(Path(path).name, 'archivo', source, target, model)
            from src.media import translate_media
            from src.asr.whisper_engine import WhisperEngine
            self.emit('status', 'Cargando reconocimiento para el archivo…')
            self.ensure_engine(model)
            count = 0
            for caption in translate_media(path, self.engine, self.translator, source, target,
                    self.stopped, lambda text: self.emit('status', text)):
                self.store.add(session, caption)
                self.emit('file_caption', caption)
                count += 1
            self.emit('status', f'Cancelado: {count} subtítulos parciales.' if self.stopped.is_set() else f'Archivo terminado: {count} subtítulos. Ya podés exportarlos.')
        except Exception as exc:
            state = 'error'
            self.emit('error', str(exc))
        finally:
            if session:
                try:
                    self.store.finish(session, 'cancelado' if self.stopped.is_set() else state)
                except Exception as exc:
                    self.emit('error', f'No se pudo actualizar el historial: {exc}')
            self.emit('done')

    def combo(self, parent, label, values, selected):
        frame = ttk.Frame(parent)
        frame.pack(side="left", padx=(0, 20))
        ttk.Label(frame, text=label).pack(anchor="w")
        combo = ttk.Combobox(frame, state="readonly", values=values, width=19)
        if selected >= 0:
            combo.current(selected)
        combo.pack(pady=4)
        return combo

    def load_sources(self):
        try:
            from src.audio.windows_capture import list_sources
            self.devices, default = list_sources()
            self.source["values"] = [label for _, label in self.devices]
            if hasattr(self,'pref_device'):
                self.pref_device['values'] = [label for _,label in self.devices]
            if self.devices:
                self.source.current(next((i for i, (index, _) in enumerate(self.devices) if index == default), 0))
                if hasattr(self,'pref_device'):
                    self.pref_device.set(self.source.get())
            else:
                self.status.set("No se encontraron dispositivos WASAPI. Conectá auriculares o altavoces y actualizá.")
        except Exception as exc:
            self.devices = []
            self.source.set('')
            self.source['values'] = []
            self.status.set(f"No se pudo acceder al audio: {exc}. Revisá la instalación de requirements.txt.")

    def emit(self, kind, value=None):
        self.events.put((kind, value))

    def launch(self, preparation):
        if self.closing or (self.worker and self.worker.is_alive()):
            return
        if not preparation and (not self.devices or self.source.current() < 0):
            messagebox.showerror("Fuente de audio", "Elegí un dispositivo de audio disponible.")
            return
        reverse = {name: code for code, name in LANGUAGES.items()}
        source_lang = reverse.get(self.origin.get())
        target_lang = reverse[self.target.get()]
        from src.asr.manager import sanitize_model_for_language
        model_size = sanitize_model_for_language(self.model.get(), source_lang)
        device = self.devices[self.source.current()][0] if self.devices else None
        self.stopped.clear()
        self.snapshot_processing()
        if not preparation:
            self.original.set("Esperando audio…")
            self.subtitle.set("Esperando la primera frase…")
            self.dialogue_manager.clear()
            self.show_overlay()
        for control in self.controls:
            control.configure(state="disabled")
        self.stop.configure(state="normal")
        self.worker = threading.Thread(target=self.run,
            args=(preparation, device, source_lang, target_lang, model_size), daemon=True)
        self.worker.start()

    def run(self, preparation, device, source_lang, target_lang, model_size):
        session = None
        state = 'completado'
        try:
            if preparation:
                prepare_models(lambda text: self.emit("status", text), self.stopped)
                if not self.stopped.is_set():
                    self.emit("status", "Verificando modelos y preparando segmentación de frases…")
                    translator = self.translator
                    samples = {"es": "Hola.", "en": "Hello.", "pt": "Olá."}
                    for source, text in samples.items():
                        for target in LANGUAGES:
                            if self.stopped.is_set():
                                return
                            translator.translate(text, source, target)
                    if not self.stopped.is_set():
                        self.emit('status', f'Preparando modelo {model_size}…')
                        self.ensure_engine(model_size)
                self.emit("status", "Preparación detenida." if self.stopped.is_set() else "Idiomas listos: español, inglés y portugués. Ya podés iniciar.")
                return
            session = self.store.create('Traducción en vivo', 'en vivo', source_lang, target_lang, model_size)
            from src.audio.windows_capture import WindowsCapture

            vad_prof = getattr(self.preferences, 'vad_profile', 'natural')
            sep_mode = getattr(self.preferences, 'separation_mode', 'auto')

            self.capture = WindowsCapture(
                device,
                threshold=self.active_threshold,
                chunk_seconds=self.active_chunk,
                vad_profile=vad_prof,
            )
            self.capture.start()
            self.emit("status", "Capturando audio. Cargando reconocimiento; las primeras frases quedan en memoria…")
            self.ensure_engine(model_size)
            if self.stopped.is_set():
                return
            translator = self.translator
            self.emit("status", "Reconocimiento listo. Procesando el audio capturado en orden…")
            prompt = translator.get_whisper_prompt() if hasattr(translator, "get_whisper_prompt") and source_lang in (None, 'en') else None

            # The active live path translates only ConfirmedUtterance objects. Capture
            # has already been running while models load, so its queued audio is kept.
            self._run_confirmed_live_pipeline(
                session=session,
                source_lang=source_lang,
                target_lang=target_lang,
                prompt=prompt,
                separation_mode=sep_mode,
            )
            return

        except Exception as exc:
            state = 'error'
            logger.exception("Error en el bucle principal de traducción: %s", exc)
            self.emit("error", str(exc))
        finally:
            if self.capture:
                self.capture.stop()
                self.capture = None
            if session:
                try:
                    self.store.finish(session, 'detenido' if self.stopped.is_set() else state)
                except Exception as exc:
                    self.emit('error', f'No se pudo actualizar el historial: {exc}')
            self.emit("done")

    def _run_confirmed_live_pipeline(self, session, source_lang, target_lang, prompt, separation_mode):
        """Run typed asynchronous capture -> ASR -> confirmation -> translation."""
        from src.pipeline.controller import PipelineController
        from src.speakers.overlap_detector import OverlapDetector
        from src.separation.manager import SpeechSeparatorManager
        from src.speakers.tracker import SpeakerTracker
        from src.translation.context_manager import ConversationContextManager
        from src.translation.context_aware_translator import ContextAwareTranslator
        from src.utils.debug_logger import SubtitleDebugLogger

        context_mgr = ConversationContextManager()
        context_translator = ContextAwareTranslator(
            glossary=self.translator.glossary if hasattr(self.translator, "glossary") else None,
            context_manager=context_mgr,
        )
        debug_logger = SubtitleDebugLogger(enabled=True)
        overlap_detector = None if separation_mode == "disabled" else OverlapDetector(sample_rate=16000)
        separator_manager = None if separation_mode == "disabled" else SpeechSeparatorManager(mode=separation_mode, sample_rate=16000)
        last_status = [0.0]

        def on_debug(stage, speaker_id, data):
            debug_logger.log_stage(stage, speaker_id, data.get("text", ""),
                                   reason=data.get("reason", ""), score=data.get("score", ""))

        def on_subtitle(item):
            translated_plain = " ".join(item.translated.split())
            self.store.add(session, Caption(
                item.start_time,
                item.end_time,
                item.original,
                translated_plain,
                item.source_language,
                speaker_id=item.speaker_id,
            ))
            self.emit("subtitle", (
                item.original,
                item.translated,
                item.source_language,
                item.speaker_id,
                item.is_overlap,
            ))
            debug_logger.log_event(
                raw_stt=item.partial_fragments[0] if item.partial_fragments else item.original,
                merged_stt=item.original,
                context_segments=context_mgr.get_speaker_context(item.speaker_id),
                translation=translated_plain,
                speaker_id=item.speaker_id,
                stt_confidence=1.0,
                asr_latency_ms=item.asr_latency * 1000.0,
                translation_latency_ms=item.trans_latency * 1000.0,
                total_latency_ms=item.total_latency * 1000.0,
                final_asr_pass_used=item.final_asr_pass_used,
                completion_reason=item.completion_reason,
            )

        controller = PipelineController(
            audio_capture=self.capture,
            vad=None,
            asr_engine=self.engine,
            translator=context_translator,
            source_language=source_lang,
            target_language=target_lang,
            on_subtitle=on_subtitle,
            session_id=session,
            max_queue_size=80,
            overlap_detector=overlap_detector,
            separator_manager=separator_manager,
            speaker_tracker=SpeakerTracker(sample_rate=16000),
            presegmented_queue=self.capture.segments,
            on_pipeline_debug=on_debug,
        )
        self.pipeline_controller = controller
        try:
            controller.start(capture_already_running=True)
            self.emit("status", "Escuchando · los parciales se acumulan; sólo se traducen oraciones confirmadas.")
            while not self.stopped.is_set():
                if self.capture.error and self.capture.segments.empty():
                    raise self.capture.error
                now = time.monotonic()
                if now - last_status[0] >= 0.5:
                    metrics = controller.get_metrics()
                    self.emit(
                        "status",
                        f"Escuchando · confirmadas: {metrics.total_segments_processed} · "
                        f"audio pendiente: {self.capture.pending_seconds:.1f} s · "
                        f"ASR: {metrics.last_asr_latency_ms:.0f} ms · traducción: {metrics.last_trans_latency_ms:.0f} ms",
                    )
                    last_status[0] = now
                time.sleep(0.05)
        finally:
            controller.stop()
            self.pipeline_controller = None

    def request_stop(self):
        self.stopped.set()
        if self.capture:
            self.capture.stopped.set()
        self.status.set("Deteniendo… Esperando que termine la operación en curso.")
        self.stop.configure(state="disabled")
        self.cancel_file.configure(state='disabled')

    def poll(self):
        for _ in range(100):
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "subtitle":
                if len(value) == 5:
                    original, translated, language, spk_id, is_ov = value
                else:
                    original, translated, language = value[:3]
                    spk_id, is_ov = None, False

                self.dialogue_manager.add_turn(original, translated, language, speaker_id=spk_id, is_overlap=is_ov)
                display_trans = self.dialogue_manager.get_display_translated()
                display_orig = self.dialogue_manager.get_display_original()
                self.original.set(f"{LANGUAGES[language]}: {display_orig}" if len(self.dialogue_manager.turns) <= 1 else display_orig)
                self.subtitle.set(display_trans)
                self.caption_updated = time.monotonic()
                self.history.configure(state="normal")
                spk_prefix = f"[{spk_id}] " if spk_id else ""
                self.history.insert("end", f"{spk_prefix}{original}\n→ {spk_prefix}{translated}\n\n")
                if int(self.history.index("end-1c").split(".")[0]) > 400:
                    self.history.delete("1.0", "100.0")
                self.history.see("end")
                self.history.configure(state="disabled")
            elif kind == 'file_caption':
                self.file_captions.append(value)
                self.file_preview.configure(state='normal')
                self.file_preview.insert('end', f'[{value.start:.2f} – {value.end:.2f}] {value.original}\n→ {value.translated}\n\n')
                self.file_preview.see('end')
                self.file_preview.configure(state='disabled')
            elif kind in ("status", "error"):
                self.status.set(value)
                if kind == "error":
                    messagebox.showerror("No se pudo continuar", value)
            elif kind == "done":
                self.reload_history()
                self.model_inventory()
                for control in self.controls:
                    control.configure(state="readonly" if isinstance(control, ttk.Combobox) else "normal")
                self.stop.configure(state="disabled")
                self.cancel_file.configure(state='disabled')
                if self.stopped.is_set():
                    self.status.set("Detenido. Los resultados parciales siguen disponibles.")
        self.meter["value"] = self.capture.level if self.capture else 0
        if self.capture and self.caption_updated and time.monotonic() - self.caption_updated > 15:
            self.subtitle.set("")
            self.original.set("")
            self.dialogue_manager.clear()
            self.caption_updated = 0
        self.root.after(100, self.poll)

    def toggle_overlay(self):
        if self.overlay and self.overlay.exists():
            if self.overlay.window.state() == "withdrawn":
                self.overlay.window.deiconify()
                self.overlay.window.lift()
            else:
                self.overlay.window.withdraw()
        else:
            self.show_overlay()

    def show_overlay(self):
        if self.overlay and self.overlay.exists():
            if self.overlay.window.state() == "withdrawn":
                self.overlay.window.deiconify()
            self.overlay.window.lift()
            return
        from src.ui.overlay import SubtitleOverlay
        self.overlay = SubtitleOverlay(self.root, self.subtitle, self.original,
            font_size=self.preferences.font_size, opacity=self.preferences.opacity)

    def close(self):
        self.closing = True
        self.stopped.set()
        if self.capture:
            self.capture.stopped.set()
        if self.worker and self.worker.is_alive():
            self.status.set('Cerrando… Esperando que termine la operación en curso y se guarde el historial.')
            self.root.after(100,self.close)
        else:
            self.root.destroy()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    TranslatorApp(tk.Tk()).root.mainloop()

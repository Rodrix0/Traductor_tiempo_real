"""Model catalogue and validated preference controls for the desktop app."""
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from src.preferences import Preferences, save_preferences
from src.translation.local import LANGUAGES


def configure_theme(root):
    root.configure(bg='#10141d')
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', background='#10141d', foreground='#e4eaf4', font=('Segoe UI', 10))
    style.configure('TNotebook', background='#10141d', borderwidth=0)
    style.configure('TNotebook.Tab', padding=(18,12), background='#1b2432')
    style.map('TNotebook.Tab', background=[('selected','#294971')])
    style.configure('TButton', background='#26374e', padding=(10,7))
    style.map('TButton', background=[('active','#36577f')], foreground=[('disabled','#718097')])
    style.configure('TCombobox', fieldbackground='#202b3a', background='#26374e', arrowcolor='#c5d7f0')
    style.map('TCombobox', fieldbackground=[('readonly','#202b3a'),('disabled','#171e29')], foreground=[('readonly','#e4eaf4'),('disabled','#718097')])
    style.configure('TEntry', fieldbackground='#202b3a', insertcolor='white')
    style.configure('TSpinbox', fieldbackground='#202b3a', background='#26374e', foreground='#e4eaf4', arrowcolor='#c5d7f0', insertcolor='white')
    style.map('TEntry', fieldbackground=[('readonly','#202b3a'),('disabled','#171e29')], foreground=[('disabled','#718097')])
    style.configure('Treeview', background='#171f2c', fieldbackground='#171f2c', rowheight=30)
    style.map('Treeview', background=[('selected','#294971')])
    style.configure('Treeview.Heading', background='#26374e', padding=6)
    style.configure('Horizontal.TProgressbar', background='#6fb7ee', troughcolor='#1b2432')
    root.option_add('*TCombobox*Listbox.background', '#202b3a')
    root.option_add('*TCombobox*Listbox.foreground', '#e4eaf4')


def style_text_widgets(parent):
    for widget in parent.winfo_children():
        if isinstance(widget, tk.Text):
            widget.configure(bg='#171f2c', fg='#e4eaf4', insertbackground='white', relief='flat', padx=12, pady=10)
        style_text_widgets(widget)


class SettingsPanel:
    def build_settings_tab(self):
        page = ttk.Frame(self.tabs, padding=20)
        self.tabs.add(page, text='  Modelos y ajustes  ')
        ttk.Label(page, text='A tu manera', font=('Segoe UI',22,'bold')).pack(anchor='w')
        ttk.Label(page, text='Preferencias iniciales para En vivo y Archivos. Guardá los cambios para aplicarlos.').pack(anchor='w', pady=8)
        row = ttk.Frame(page)
        row.pack(fill='x', pady=10)
        self.pref_origin = self.combo(row, 'Idioma del audio', ['Automático'] + list(LANGUAGES.values()), 0)
        self.pref_target = self.combo(row, 'Traducir a', list(LANGUAGES.values()), 0)
        self.pref_model = self.combo(row, 'Calidad', ['base','small'], 0)
        ttk.Label(page, text='base: más rápido en CPU. small: mayor precisión y más tiempo de procesamiento.').pack(anchor='w')
        row = ttk.Frame(page)
        row.pack(fill='x', pady=10)
        self.pref_compute = self.combo(row, 'Procesamiento', ['Automático','CPU'], 0)
        self.pref_device = self.combo(row, 'Dispositivo de audio', [], -1)
        self.pref_device.configure(width=60)
        ttk.Label(page, text='Automático intenta usar GPU y recurre a CPU si faltan componentes.').pack(anchor='w')
        row = ttk.Frame(page)
        row.pack(fill='x', pady=14)
        self.pref_font = self.number_field(row, 'Tamaño de subtítulos (16–48)', 24)
        self.pref_opacity = self.number_field(row, 'Opacidad (0.45–1)', 0.95)
        row = ttk.Frame(page)
        row.pack(fill='x', pady=8)
        self.pref_threshold = self.number_field(row, 'Umbral de voz (0.002–0.05)', 0.008)
        self.pref_chunk = self.number_field(row, 'Fragmento máximo (2–8 s)', 4)
        ttk.Label(page, text='Bajá el umbral para voces suaves. Los fragmentos más cortos reducen la espera, pero pueden cortar frases.').pack(anchor='w', pady=6)
        row = ttk.Frame(page)
        row.pack(fill='x', pady=12)
        self.save_settings = ttk.Button(row, text='Guardar preferencias', command=self.save_settings_action)
        self.save_settings.pack(side='left')
        self.prepare_settings = ttk.Button(row, text='Preparar modelo e idiomas', command=self.prepare_from_settings)
        self.prepare_settings.pack(side='left', padx=8)
        ttk.Button(row, text='Ver modelos instalados', command=self.model_inventory).pack(side='left')
        self.inventory = tk.StringVar()
        ttk.Label(page, textvariable=self.inventory, wraplength=900).pack(anchor='w', pady=8)
        ttk.Label(page, textvariable=self.status, wraplength=900).pack(anchor='w', pady=8)
        self.controls.extend([self.pref_origin,self.pref_target,self.pref_model,self.pref_compute,self.pref_device,
            self.pref_font,self.pref_opacity,self.pref_threshold,self.pref_chunk,self.save_settings,self.prepare_settings])
        self.model_inventory()

    def number_field(self, parent, label, value):
        frame = ttk.Frame(parent)
        frame.pack(side='left', padx=(0,24))
        ttk.Label(frame,text=label).pack(anchor='w')
        entry = ttk.Entry(frame, width=25)
        entry.insert(0,str(value))
        entry.pack(anchor='w',pady=5)
        return entry

    def model_inventory(self):
        root = Path(__file__).resolve().parents[2] / 'models'
        states = []
        for model in ('base','small'):
            snapshots = root / f'models--Systran--faster-whisper-{model}' / 'snapshots'
            installed = any(all((p/name).is_file() for name in ('model.bin','config.json','tokenizer.json')) for p in snapshots.glob('*'))
            states.append(f'{model}: ' + ('descargado' if installed else 'sin descargar'))
        try:
            import argostranslate.package as package
            from src.translation.local import PAIRS
            installed = {(p.from_code,p.to_code) for p in package.get_installed_packages()}
            states.append(f'Traducción: {sum(pair in installed for pair in PAIRS)}/4 modelos')
        except Exception as exc:
            states.append(f'Traducción no disponible: {exc}')
        if self.engine:
            states.append(f'Motor cargado: {self.engine_size[0]} · {self.engine.actual_device.upper()}')
        if self.translator._marian_available():
            states.append('Inglés → español: Marian (local)')
        self.inventory.set('   |   '.join(states))

    def apply_preferences(self):
        p = self.preferences
        origin = LANGUAGES.get(p.source, 'Automático')
        for control in (self.origin,self.file_origin,self.pref_origin):
            control.set(origin)
        for control in (self.target,self.file_target,self.pref_target):
            control.set(LANGUAGES[p.target])
        for control in (self.model,self.file_model,self.pref_model):
            control.set(p.model)
        self.pref_compute.set('CPU' if p.compute == 'cpu' else 'Automático')
        for control, value in [(self.pref_font,p.font_size),(self.pref_opacity,p.opacity),
                (self.pref_threshold,p.threshold),(self.pref_chunk,p.chunk_seconds)]:
            control.delete(0,'end')
            control.insert(0,str(value))
        labels = [label for _,label in self.devices]
        if p.device_name in labels:
            self.source.current(labels.index(p.device_name))
        self.pref_device.set(self.source.get())
        if self.overlay and self.overlay.exists():
            self.overlay.font_size.set(p.font_size)
            self.overlay.opacity.set(p.opacity)
            self.overlay.refresh()

    def save_settings_action(self):
        reverse = {name: code for code,name in LANGUAGES.items()}
        try:
            preferences = Preferences(source=reverse.get(self.pref_origin.get(),'auto'), target=reverse[self.pref_target.get()],
                model=self.pref_model.get(), compute='cpu' if self.pref_compute.get()=='CPU' else 'auto',
                device_name=self.pref_device.get(), font_size=int(self.pref_font.get()),
                opacity=float(self.pref_opacity.get()), threshold=float(self.pref_threshold.get()), chunk_seconds=float(self.pref_chunk.get()))
            save_preferences(self.data_dir/'preferences.json', preferences)
            self.preferences = preferences
            self.apply_preferences()
            self.status.set('Preferencias guardadas. Se aplican al próximo inicio.')
            return True
        except (OSError,ValueError,KeyError) as exc:
            messagebox.showerror('Revisá los ajustes',str(exc))
            return False

    def prepare_from_settings(self):
        if self.save_settings_action():
            self.launch(True)

    def snapshot_processing(self):
        self.active_compute = self.preferences.compute
        self.active_threshold = self.preferences.threshold
        self.active_chunk = self.preferences.chunk_seconds

    def ensure_engine(self, model):
        key = (model,self.active_compute)
        if self.engine is None or self.engine_size != key:
            from src.asr.whisper_engine import WhisperEngine
            self.engine = None
            self.engine = WhisperEngine(model_size=model,device=self.active_compute)
            self.engine_size = key

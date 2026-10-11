from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import tkinter as tk
import wave
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

from audio_transcriber import (
    AUDIO_EXTENSIONS,
    TranscriptionDependencyError,
    list_audio_files,
    transcribe_audio_files,
    transcribe_audio_path,
)
from hackathon import KeywordHit, build_keyword_terms_from_data, find_hits
from hackathon import load_synonym_data, term_is_direct


TRANSCRIPT_DIR = Path("Transkript")
DEFAULT_AUDIO_DIR = Path("Audio")
DEFAULT_KEYWORDS = Path("Stichwortliste.json")
DEFAULT_SYNONYMS = Path("Synonymliste.example.json")
CURRENT_TRANSCRIPT = TRANSCRIPT_DIR / "_current_transcribed_text.txt"
BUNDLED_MODEL_DIR = Path("models")
TIMESTAMP_RE = re.compile(r"\[(\d{2}:\d{2}(?::\d{2})?\.\d{3})\s*-\s*(\d{2}:\d{2}(?::\d{2})?\.\d{3})\]")

TYPE_STYLES = {
    "exact": {"label": "EXACT", "foreground": "#b91c1c", "background": "#fee2e2"},
    "synonym": {"label": "SYNONYM", "foreground": "#1d4ed8", "background": "#dbeafe"},
    "context": {"label": "CONTEXT", "foreground": "#a16207", "background": "#fef3c7"},
}

COLORS = {
    "bg": ("#f5f7fb", "#0f172a"),
    "sidebar": ("#eef2f7", "#111827"),
    "card": ("#ffffff", "#1e293b"),
    "text": ("#0f172a", "#f8fafc"),
    "muted": ("#64748b", "#cbd5e1"),
    "border": ("#cbd5e1", "#334155"),
    "subtle": ("#e2e8f0", "#334155"),
    "subtle_hover": ("#cbd5e1", "#475569"),
    "primary": ("#2563eb", "#3b82f6"),
    "primary_hover": ("#1d4ed8", "#2563eb"),
    "graph_bg": ("#ffffff", "#1e293b"),
    "axis": ("#cbd5e1", "#475569"),
}


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent


class TranscriptScannerApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.title("Swiss German Fraud Call Scanner")
        self.geometry("1280x820")
        self.minsize(1040, 680)

        self.app_dir = app_base_dir()
        self.resource_dir = resource_base_dir()
        self.keyword_path = tk.StringVar(value=str(self.resource_dir / DEFAULT_KEYWORDS))
        self.audio_dir = tk.StringVar(value=str(self.app_dir / DEFAULT_AUDIO_DIR))
        self.audio_output_dir = tk.StringVar(value=str(self.app_dir / TRANSCRIPT_DIR))
        self.audio_model = tk.StringVar(value="tiny")
        self.audio_language = tk.StringVar(value="de")
        self.language = tk.StringVar(value="all")
        self.min_confidence = tk.DoubleVar(value=0.70)
        self.confidence_label = tk.StringVar()
        self.context_window = tk.IntVar(value=2)
        self.accent_insensitive = tk.BooleanVar(value=True)
        self.direct_only = tk.BooleanVar(value=False)
        self.theme_mode = tk.StringVar(value="Light")
        self.status = tk.StringVar(value="Ready")

        self.hits: list[KeywordHit] = []
        self.generated_transcript_paths: list[Path] = []
        self.selected_audio_files: list[Path] = []
        self.current_transcript_text = ""
        self.playing_audio: Path | None = None
        self.audio_duration = 0.0
        self.play_start_offset = 0.0
        self.play_started_at = 0.0
        self.is_audio_playing = False
        self.is_audio_paused = False
        self.playhead_after_id: str | None = None
        self.audio_sound = None
        self.audio_channel = None

        self.audio_ready = True
        try:
            pygame.mixer.init()
        except pygame.error:
            self.audio_ready = False

        self.configure(fg_color=COLORS["bg"])
        self.min_confidence.trace_add("write", self.on_confidence_change)
        self.update_confidence_label()
        self.create_widgets()

    def create_widgets(self) -> None:
        self.grid_columnconfigure(0, minsize=330)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkScrollableFrame(
            self,
            width=300,
            fg_color=COLORS["sidebar"],
            corner_radius=0,
        )
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_columnconfigure(0, weight=1)

        content = ctk.CTkFrame(self, fg_color=COLORS["bg"], corner_radius=0)
        content.grid(row=0, column=1, sticky="nsew", padx=(18, 20), pady=18)
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(3, weight=1)

        self.build_sidebar(sidebar)
        self.build_content(content)

    def build_sidebar(self, sidebar: ctk.CTkScrollableFrame) -> None:
        ctk.CTkLabel(
            sidebar,
            text="Fraud Call Scanner",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=COLORS["text"],
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(18, 2))
        ctk.CTkLabel(
            sidebar,
            text="Audio transcription and keyword analysis",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["muted"],
        ).grid(row=1, column=0, sticky="w", padx=16, pady=(0, 18))

        ctk.CTkSegmentedButton(
            sidebar,
            values=["Light", "Night"],
            variable=self.theme_mode,
            command=self.change_theme,
            height=34,
            corner_radius=8,
            fg_color=COLORS["subtle"],
            selected_color=COLORS["primary"],
            selected_hover_color=COLORS["primary_hover"],
            unselected_color=COLORS["subtle"],
            unselected_hover_color=COLORS["subtle_hover"],
            text_color=COLORS["text"],
        ).grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))

        keyword_card = self.sidebar_card(sidebar, 3, "Keyword Source")
        self.path_row(keyword_card, 1, self.keyword_path, "Browse JSON", self.choose_keyword_json)

        audio_card = self.sidebar_card(sidebar, 4, "Audio Input")
        self.path_row(audio_card, 1, self.audio_dir, "Audio File", self.choose_audio_path)
        ctk.CTkButton(
            audio_card,
            text="Multiple Files",
            command=self.choose_audio_files,
            height=34,
            corner_radius=8,
            fg_color=COLORS["subtle"],
            hover_color=COLORS["subtle_hover"],
            text_color=COLORS["text"],
        ).grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 8))
        ctk.CTkButton(
            audio_card,
            text="Audio Folder",
            command=self.choose_audio_folder,
            height=34,
            corner_radius=8,
            fg_color=COLORS["subtle"],
            hover_color=COLORS["subtle_hover"],
            text_color=COLORS["text"],
        ).grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 12))

        speech_card = self.sidebar_card(sidebar, 5, "Speech Settings")
        speech_card.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkLabel(speech_card, text="Model", text_color=COLORS["muted"]).grid(
            row=1, column=0, sticky="w", padx=(12, 6), pady=(0, 4)
        )
        ctk.CTkLabel(speech_card, text="Language", text_color=COLORS["muted"]).grid(
            row=1, column=1, sticky="w", padx=(6, 12), pady=(0, 4)
        )
        ctk.CTkOptionMenu(
            speech_card,
            values=["tiny", "base", "small", "medium"],
            variable=self.audio_model,
            height=34,
            corner_radius=8,
        ).grid(row=2, column=0, sticky="ew", padx=(12, 6), pady=(0, 12))
        ctk.CTkOptionMenu(
            speech_card,
            values=["de", "auto"],
            variable=self.audio_language,
            height=34,
            corner_radius=8,
        ).grid(row=2, column=1, sticky="ew", padx=(6, 12), pady=(0, 12))

        analysis_card = self.sidebar_card(sidebar, 6, "Analysis Settings")
        analysis_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(analysis_card, text="Keyword Language", text_color=COLORS["muted"]).grid(
            row=1, column=0, sticky="w", padx=12, pady=(0, 4)
        )
        ctk.CTkOptionMenu(
            analysis_card,
            values=["all", "gsw", "de"],
            variable=self.language,
            height=34,
            corner_radius=8,
        ).grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))

        confidence_row = ctk.CTkFrame(analysis_card, fg_color="transparent")
        confidence_row.grid(row=3, column=0, sticky="ew", padx=12)
        confidence_row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(confidence_row, text="Minimum Confidence", text_color=COLORS["muted"]).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(confidence_row, textvariable=self.confidence_label, text_color=COLORS["primary"]).grid(row=0, column=1, sticky="e")
        ctk.CTkSlider(
            analysis_card,
            from_=0,
            to=1,
            variable=self.min_confidence,
            command=lambda _value: self.update_confidence_label(),
        ).grid(row=4, column=0, sticky="ew", padx=12, pady=(6, 12))

        turns_row = ctk.CTkFrame(analysis_card, fg_color="transparent")
        turns_row.grid(row=5, column=0, sticky="ew", padx=12, pady=(0, 12))
        turns_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(turns_row, text="Nearby Turns", text_color=COLORS["muted"]).grid(row=0, column=0, sticky="w")
        ctk.CTkOptionMenu(
            turns_row,
            values=["0", "1", "2", "3", "4", "5"],
            command=lambda value: self.context_window.set(int(value)),
            height=32,
            width=86,
            corner_radius=8,
        ).grid(row=0, column=1, sticky="e")
        self.context_window.set(2)

        ctk.CTkSwitch(
            analysis_card,
            text="Accent-insensitive",
            variable=self.accent_insensitive,
            progress_color=COLORS["primary"],
        ).grid(row=6, column=0, sticky="w", padx=12, pady=(0, 8))
        ctk.CTkSwitch(
            analysis_card,
            text="Direct only",
            variable=self.direct_only,
            progress_color=COLORS["primary"],
        ).grid(row=7, column=0, sticky="w", padx=12, pady=(0, 14))

        self.transcribe_button = ctk.CTkButton(
            sidebar,
            text="Transcribe + Analyse",
            command=self.transcribe_audio,
            height=44,
            corner_radius=8,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.transcribe_button.grid(row=7, column=0, sticky="ew", padx=16, pady=(8, 10))

        self.analyse_button = ctk.CTkButton(
            sidebar,
            text="Analyse Transcript",
            command=self.scan,
            height=38,
            corner_radius=8,
            fg_color=("#0f172a", "#e2e8f0"),
            hover_color=("#1e293b", "#cbd5e1"),
            text_color=("#ffffff", "#0f172a"),
        )
        self.analyse_button.grid(row=8, column=0, sticky="ew", padx=16, pady=(0, 10))

        ctk.CTkButton(
            sidebar,
            text="Export Results JSON",
            command=self.export_results,
            height=38,
            corner_radius=8,
            fg_color=COLORS["subtle"],
            hover_color=COLORS["subtle_hover"],
            text_color=COLORS["text"],
        ).grid(row=9, column=0, sticky="ew", padx=16, pady=(0, 18))

        ctk.CTkButton(
            sidebar,
            text="Refresh",
            command=self.refresh_outputs,
            height=38,
            corner_radius=8,
            fg_color=COLORS["subtle"],
            hover_color=COLORS["subtle_hover"],
            text_color=COLORS["text"],
        ).grid(row=10, column=0, sticky="ew", padx=16, pady=(0, 18))

    def build_content(self, content: ctk.CTkFrame) -> None:
        header = ctk.CTkFrame(content, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Results",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=COLORS["text"],
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            textvariable=self.status,
            font=ctk.CTkFont(size=13),
            text_color=COLORS["muted"],
        ).grid(row=0, column=1, sticky="e")
        self.transcription_progress = ctk.CTkProgressBar(
            header,
            width=180,
            height=8,
            mode="indeterminate",
            progress_color=COLORS["primary"],
        )
        self.transcription_progress.grid(row=1, column=1, sticky="e", pady=(8, 0))
        self.transcription_progress.set(0)

        summary = ctk.CTkFrame(content, fg_color="transparent")
        summary.grid(row=1, column=0, sticky="ew", pady=(16, 16))
        summary.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.total_label = self.summary_tile(summary, 0, "Total", "0", COLORS["text"])
        self.exact_label = self.summary_tile(summary, 1, "Exact", "0", "#b91c1c")
        self.synonym_label = self.summary_tile(summary, 2, "Synonym", "0", "#1d4ed8")
        self.context_label = self.summary_tile(summary, 3, "Context", "0", "#a16207")

        self.tabs = ctk.CTkTabview(
            content,
            fg_color=COLORS["card"],
            segmented_button_fg_color=COLORS["subtle"],
            segmented_button_selected_color=COLORS["primary"],
            segmented_button_selected_hover_color=COLORS["primary_hover"],
            segmented_button_unselected_color=COLORS["subtle"],
            segmented_button_unselected_hover_color=COLORS["subtle_hover"],
            corner_radius=8,
        )
        self.tabs.grid(row=3, column=0, sticky="nsew")
        self.tabs.add("Transcribed Text")
        self.tabs.add("Fraud Findings")
        self.tabs.add("Graph")
        self.make_tab_titles_visible()

        transcript_tab = self.tabs.tab("Transcribed Text")
        findings_tab = self.tabs.tab("Fraud Findings")
        graph_tab = self.tabs.tab("Graph")
        findings_tab.grid_columnconfigure(0, weight=1)
        findings_tab.grid_rowconfigure(0, weight=1)
        transcript_tab.grid_columnconfigure(0, weight=1)
        transcript_tab.grid_rowconfigure(0, weight=1)
        graph_tab.grid_columnconfigure(0, weight=1)
        graph_tab.grid_rowconfigure(0, weight=1)

        self.results = ctk.CTkTextbox(
            findings_tab,
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=12),
            fg_color=COLORS["card"],
            text_color=COLORS["text"],
            border_color=COLORS["border"],
            border_width=1,
            corner_radius=8,
        )
        self.results.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.results.configure(state="disabled")

        self.transcript_tabs = ctk.CTkTabview(
            transcript_tab,
            fg_color=COLORS["card"],
            segmented_button_fg_color=COLORS["subtle"],
            segmented_button_selected_color=COLORS["primary"],
            segmented_button_selected_hover_color=COLORS["primary_hover"],
            segmented_button_unselected_color=COLORS["subtle"],
            segmented_button_unselected_hover_color=COLORS["subtle_hover"],
            corner_radius=8,
        )
        self.transcript_tabs.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.transcript_tabs.add("Transcript")
        self.transcript_tabs.add("Audio Timeline")
        self.make_segmented_titles_visible(self.transcript_tabs)

        transcript_text_tab = self.transcript_tabs.tab("Transcript")
        audio_timeline_tab = self.transcript_tabs.tab("Audio Timeline")
        transcript_text_tab.grid_columnconfigure(0, weight=1)
        transcript_text_tab.grid_rowconfigure(0, weight=1)
        audio_timeline_tab.grid_columnconfigure(0, weight=1)
        audio_timeline_tab.grid_rowconfigure(1, weight=1)

        self.transcribed_text = ctk.CTkTextbox(
            transcript_text_tab,
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=12),
            fg_color=COLORS["card"],
            text_color=COLORS["text"],
            border_color=COLORS["border"],
            border_width=1,
            corner_radius=8,
        )
        self.transcribed_text.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        playback_bar = ctk.CTkFrame(audio_timeline_tab, fg_color="transparent")
        playback_bar.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 0))
        playback_bar.grid_columnconfigure(3, weight=1)

        self.play_button = ctk.CTkButton(
            playback_bar,
            text="Play",
            width=82,
            height=34,
            corner_radius=8,
            command=self.play_audio,
        )
        self.play_button.grid(row=0, column=0, sticky="w", padx=(0, 8))

        self.pause_button = ctk.CTkButton(
            playback_bar,
            text="Pause",
            width=82,
            height=34,
            corner_radius=8,
            fg_color=COLORS["subtle"],
            hover_color=COLORS["subtle_hover"],
            text_color=COLORS["text"],
            command=self.pause_audio,
        )
        self.pause_button.grid(row=0, column=1, sticky="w", padx=(0, 8))

        self.stop_button = ctk.CTkButton(
            playback_bar,
            text="Stop",
            width=82,
            height=34,
            corner_radius=8,
            fg_color=COLORS["subtle"],
            hover_color=COLORS["subtle_hover"],
            text_color=COLORS["text"],
            command=self.stop_audio,
        )
        self.stop_button.grid(row=0, column=2, sticky="w", padx=(0, 12))

        self.audio_time_label = ctk.CTkLabel(playback_bar, text="00:00 / 00:00", text_color=COLORS["muted"])
        self.audio_time_label.grid(row=0, column=3, sticky="e")

        self.audio_playback_status = ctk.CTkLabel(playback_bar, text="", text_color=COLORS["muted"])
        self.audio_playback_status.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 0))

        self.audio_timeline = tk.Canvas(
            audio_timeline_tab,
            bg=self.theme_value("graph_bg"),
            highlightthickness=1,
            highlightbackground=self.theme_value("border"),
        )
        self.audio_timeline.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)
        self.audio_timeline.bind("<Configure>", lambda _event: self.render_audio_timeline())
        self.audio_timeline.bind("<Button-1>", self.seek_audio_from_timeline)

        self.graph_canvas = tk.Canvas(
            graph_tab,
            bg=self.theme_value("graph_bg"),
            highlightthickness=1,
            highlightbackground=self.theme_value("border"),
        )
        self.graph_canvas.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.graph_canvas.bind("<Configure>", lambda _event: self.render_graph())
        self.configure_result_tags()

    def sidebar_card(self, parent, row: int, title: str) -> ctk.CTkFrame:
        card = ctk.CTkFrame(parent, fg_color=COLORS["card"], corner_radius=8)
        card.grid(row=row, column=0, sticky="ew", padx=16, pady=(0, 12))
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["text"],
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 8))
        return card

    def path_row(self, parent, row: int, variable: tk.StringVar, button_text: str, command) -> None:
        parent.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(
            parent,
            textvariable=variable,
            height=34,
            corner_radius=8,
            border_color=COLORS["border"],
            fg_color=COLORS["card"],
            text_color=COLORS["text"],
        ).grid(row=row, column=0, sticky="ew", padx=12, pady=(0, 8))
        ctk.CTkButton(
            parent,
            text=button_text,
            command=command,
            height=34,
            corner_radius=8,
            fg_color=COLORS["subtle"],
            hover_color=COLORS["subtle_hover"],
            text_color=COLORS["text"],
        ).grid(row=row + 1, column=0, sticky="ew", padx=12, pady=(0, 12))

    def summary_tile(self, parent, column: int, label: str, value: str, color: str) -> ctk.CTkLabel:
        tile = ctk.CTkFrame(parent, fg_color=COLORS["card"], corner_radius=8)
        tile.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 8, 0))
        ctk.CTkLabel(tile, text=label, text_color=COLORS["muted"], font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, sticky="w", padx=14, pady=(12, 0)
        )
        value_label = ctk.CTkLabel(tile, text=value, text_color=color, font=ctk.CTkFont(size=22, weight="bold"))
        value_label.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 12))
        return value_label

    def configure_result_tags(self) -> None:
        self.results.tag_config("file", foreground=self.theme_value("text"))
        self.results.tag_config("muted", foreground=self.theme_value("muted"))
        self.results.tag_config("body", foreground=self.theme_value("text"))
        self.transcribed_text.tag_config("file", foreground=self.theme_value("text"))
        self.transcribed_text.tag_config("body", foreground=self.theme_value("text"))
        for evidence_type, style in TYPE_STYLES.items():
            self.results.tag_config(
                evidence_type,
                foreground=style["foreground"],
                background=style["background"],
            )

    def make_tab_titles_visible(self) -> None:
        self.make_segmented_titles_visible(self.tabs)

    def make_segmented_titles_visible(self, tabview) -> None:
        try:
            tabview._segmented_button.configure(
                text_color=("#0f172a", "#f8fafc"),
                selected_color=COLORS["primary"],
                selected_hover_color=COLORS["primary_hover"],
                unselected_color=COLORS["subtle"],
                unselected_hover_color=COLORS["subtle_hover"],
            )
        except AttributeError:
            pass

    def change_theme(self, mode: str) -> None:
        ctk.set_appearance_mode("dark" if mode == "Night" else "light")
        self.configure_result_tags()
        if hasattr(self, "graph_canvas"):
            self.graph_canvas.configure(
                bg=self.theme_value("graph_bg"),
                highlightbackground=self.theme_value("border"),
            )
        if hasattr(self, "audio_timeline"):
            self.audio_timeline.configure(
                bg=self.theme_value("graph_bg"),
                highlightbackground=self.theme_value("border"),
            )
        self.make_tab_titles_visible()
        if hasattr(self, "transcript_tabs"):
            self.make_segmented_titles_visible(self.transcript_tabs)
        self.render_audio_timeline()
        self.render_graph()

    def theme_value(self, color_name: str) -> str:
        color = COLORS[color_name]
        return color[1] if self.theme_mode.get() == "Night" else color[0]

    def choose_keyword_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose keyword JSON",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            initialdir=str(Path.cwd()),
        )
        if path:
            self.keyword_path.set(path)

    def choose_audio_path(self) -> None:
        extensions = " ".join(f"*{extension}" for extension in sorted(AUDIO_EXTENSIONS))
        path = filedialog.askopenfilename(
            title="Choose audio recording",
            filetypes=(("Audio files", extensions), ("All files", "*.*")),
            initialdir=str(Path.cwd()),
        )
        if path:
            self.selected_audio_files = [Path(path)]
            self.audio_dir.set(path)
            self.reset_audio_playback()

    def choose_audio_files(self) -> None:
        extensions = " ".join(f"*{extension}" for extension in sorted(AUDIO_EXTENSIONS))
        paths = filedialog.askopenfilenames(
            title="Choose audio recordings",
            filetypes=(("Audio files", extensions), ("All files", "*.*")),
            initialdir=str(Path.cwd()),
        )
        if paths:
            self.selected_audio_files = [Path(path) for path in paths]
            if len(paths) == 1:
                self.audio_dir.set(paths[0])
            else:
                self.audio_dir.set(f"{len(paths)} audio files selected")
            self.reset_audio_playback()

    def choose_audio_folder(self) -> None:
        path = filedialog.askdirectory(title="Choose audio folder", initialdir=str(Path.cwd()))
        if path:
            self.selected_audio_files = []
            self.audio_dir.set(path)
            self.reset_audio_playback()

    def selected_transcripts(self) -> list[Path]:
        return [self.write_transcript_box_for_scan()]

    def displayed_transcript_text(self) -> str:
        return self.transcribed_text.get("1.0", "end-1c").strip()

    def write_transcript_box_for_scan(self) -> Path:
        transcript_text = self.displayed_transcript_text()
        if not transcript_text:
            raise ValueError("No transcribed text is available to analyse.")

        transcript_dir = Path(self.audio_output_dir.get())
        transcript_dir.mkdir(parents=True, exist_ok=True)
        current_transcript = transcript_dir / CURRENT_TRANSCRIPT.name
        current_transcript.write_text(transcript_text, encoding="utf-8")
        self.current_transcript_text = transcript_text
        return current_transcript

    def transcribe_audio(self) -> None:
        audio_path = Path(self.audio_dir.get())
        output_dir = Path(self.audio_output_dir.get())

        if self.selected_audio_files:
            audio_files = self.selected_audio_files
        elif audio_path.exists():
            audio_files = list_audio_files(audio_path)
        else:
            messagebox.showwarning("Audio not found", f"Audio path not found: {audio_path}")
            return

        if not audio_files:
            messagebox.showwarning("No audio", "No supported audio files were found.")
            return

        self.set_busy(True, f"Transcribing {len(audio_files)} audio file(s)...")

        def worker() -> None:
            try:
                if self.selected_audio_files:
                    generated_paths = transcribe_audio_files(
                        audio_files,
                        output_dir,
                        model_name=self.selected_model_source(),
                        language=self.audio_language.get(),
                    )
                else:
                    generated_paths = transcribe_audio_path(
                        audio_path,
                        output_dir,
                        model_name=self.selected_model_source(),
                        language=self.audio_language.get(),
                    )
            except Exception as exc:
                self.after(0, lambda error=exc: self.show_transcription_error(error))
                return

            self.after(0, lambda paths=generated_paths: self.finish_transcription(paths))

        threading.Thread(target=worker, daemon=True).start()

    def selected_model_source(self) -> str:
        model_name = self.audio_model.get()
        bundled_model = self.resource_dir / BUNDLED_MODEL_DIR / f"faster-whisper-{model_name}"
        if bundled_model.exists():
            return str(bundled_model)
        return model_name

    def set_busy(self, busy: bool, status: str) -> None:
        self.status.set(status)
        self.transcribe_button.configure(
            state="disabled" if busy else "normal",
            text="Transcribing..." if busy else "Transcribe + Analyse",
        )
        if hasattr(self, "analyse_button"):
            self.analyse_button.configure(state="disabled" if busy else "normal")
        if hasattr(self, "transcription_progress"):
            if busy:
                self.transcription_progress.start()
            else:
                self.transcription_progress.stop()
                self.transcription_progress.set(0)
        self.update_idletasks()

    def show_transcription_error(self, error: Exception) -> None:
        self.set_busy(False, "Transcription failed")
        if isinstance(error, TranscriptionDependencyError):
            messagebox.showerror("Transcription setup needed", str(error))
        else:
            messagebox.showerror("Transcription failed", str(error))

    def finish_transcription(self, generated_paths: list[Path]) -> None:
        self.generated_transcript_paths = generated_paths
        self.render_transcribed_text(generated_paths)
        self.set_busy(False, f"Generated {len(generated_paths)} transcript file(s)")
        self.scan()

    def render_transcribed_text(self, generated_paths: list[Path]) -> None:
        self.transcribed_text.configure(state="normal")
        self.transcribed_text.delete("1.0", tk.END)

        if not generated_paths:
            self.current_transcript_text = ""
            self.transcribed_text.insert(tk.END, "No transcribed text yet.\n", "body")
        else:
            self.current_transcript_text = "\n\n".join(
                path.read_text(encoding="utf-8-sig").strip() for path in generated_paths
            )
            for path in generated_paths:
                self.transcribed_text.insert(tk.END, path.read_text(encoding="utf-8-sig").strip(), "body")
                self.transcribed_text.insert(tk.END, "\n\n", "body")

        self.render_audio_timeline()

    def on_confidence_change(self, *_args) -> None:
        self.update_confidence_label()

    def selected_min_confidence(self) -> float:
        try:
            value = float(self.min_confidence.get())
        except (tk.TclError, ValueError):
            value = 0.0
        return max(0.0, min(1.0, value))

    def update_confidence_label(self) -> None:
        value = self.selected_min_confidence()
        self.confidence_label.set(f"{value:.0%}")

    def scan(self) -> None:
        try:
            terms = self.load_terms()
            if self.direct_only.get():
                terms = [term for term in terms if term_is_direct(term)]

            try:
                selected_paths = self.selected_transcripts()
            except ValueError as exc:
                messagebox.showwarning("No transcript", str(exc))
                return

            self.status.set("Scanning...")
            self.update_idletasks()
            hits = find_hits(
                selected_paths,
                terms,
                accent_insensitive=self.accent_insensitive.get(),
                include_context=not self.direct_only.get(),
                include_fuzzy=not self.direct_only.get(),
                context_window=max(0, self.context_window.get()),
            )
            minimum = self.selected_min_confidence()
            self.hits = [hit for hit in hits if hit.confidence >= minimum]
            self.render_results()
            self.status.set(f"Scan complete: {len(self.hits)} hits")
        except Exception as exc:
            messagebox.showerror("Scan failed", str(exc))
            self.status.set("Scan failed")

    def load_terms(self):
        keyword_path = Path(self.keyword_path.get())
        languages = {"de", "gsw"} if self.language.get() == "all" else {self.language.get()}
        if not keyword_path.exists():
            raise FileNotFoundError(f"Keyword JSON not found: {keyword_path}")

        data = json.loads(keyword_path.read_text(encoding="utf-8-sig"))
        synonym_path = self.resource_dir / DEFAULT_SYNONYMS
        synonym_data = load_synonym_data(synonym_path if synonym_path.exists() else None)
        return build_keyword_terms_from_data(data, languages, synonym_data=synonym_data)

    def render_results(self) -> None:
        self.results.configure(state="normal")
        self.results.delete("1.0", tk.END)

        if not self.hits:
            self.results.insert(tk.END, "No matches found.\n", "muted")
        else:
            current_file = None
            for hit in self.hits:
                if hit.file != current_file:
                    current_file = hit.file
                    self.results.insert(tk.END, f"\n{Path(hit.file).name}\n", "file")
                self.insert_hit(hit)

        self.results.configure(state="disabled")
        self.update_summary()
        self.render_audio_timeline()
        self.render_graph()

    def insert_hit(self, hit: KeywordHit) -> None:
        style = TYPE_STYLES.get(hit.evidence_type, TYPE_STYLES["exact"])
        speaker = f" - {hit.speaker}" if hit.speaker else ""
        header = (
            f"{style['label']}  {hit.confidence:.2f}  "
            f"line {hit.line_number}, {hit.turn_id}{speaker}  "
            f"{hit.keyword_id}: {hit.keyword_label}\n"
        )
        self.results.insert(tk.END, header, hit.evidence_type)
        self.results.insert(tk.END, f"Matched: {hit.matched_text}\n", "muted")
        self.results.insert(tk.END, f"Reason: {hit.reason}\n", "muted")
        self.results.insert(tk.END, f"{hit.text}\n", "body")

    def update_summary(self) -> None:
        exact = self.count_hits("exact")
        synonym = self.count_hits("synonym")
        context = self.count_hits("context")
        self.total_label.configure(text=str(len(self.hits)))
        self.exact_label.configure(text=str(exact))
        self.synonym_label.configure(text=str(synonym))
        self.context_label.configure(text=str(context))
        self.status.set(f"Scan complete: {len(self.hits)} hits")

    def render_audio_timeline(self) -> None:
        if not hasattr(self, "audio_timeline"):
            return

        transcript_text = self.displayed_transcript_text() if hasattr(self, "transcribed_text") else ""
        canvas = self.audio_timeline
        canvas.configure(bg=self.theme_value("graph_bg"), highlightbackground=self.theme_value("border"))
        canvas.delete("all")

        width = max(canvas.winfo_width(), 500)
        height = max(canvas.winfo_height(), 260)
        text_color = self.theme_value("text")
        muted = self.theme_value("muted")
        axis = self.theme_value("axis")

        canvas.create_text(
            24,
            24,
            text="Audio Timestamp Highlights",
            anchor="w",
            fill=text_color,
            font=("Segoe UI", 16, "bold"),
        )

        if not transcript_text:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Transcribe an audio file to see highlighted timestamp ranges.",
                fill=muted,
                font=("Segoe UI", 12),
            )
            return

        ranges = self.transcript_timestamp_ranges(transcript_text)
        if not ranges:
            canvas.create_text(
                width / 2,
                height / 2,
                text="No timestamps found in the current transcript.",
                fill=muted,
                font=("Segoe UI", 12),
            )
            return

        audio_duration = self.current_audio_duration()
        duration = audio_duration or max(end for _line_number, _start, end in ranges)
        duration = max(duration, 1.0)
        self.audio_duration = duration
        hit_ranges = self.hit_timestamp_ranges(ranges)

        left = 72
        right = width - 42
        bar_y = 118
        bar_height = 26
        usable_width = max(right - left, 120)
        self.timeline_geometry = (left, right, bar_y, bar_height, duration)

        audio_names = self.current_audio_names()
        audio_label = ", ".join(audio_names[:2])
        if len(audio_names) > 2:
            audio_label += f" + {len(audio_names) - 2} more"
        if len(audio_names) > 1:
            audio_label += " (playing first selected file)"
        canvas.create_text(left, 66, text=audio_label or "Selected audio", anchor="w", fill=muted, font=("Segoe UI", 11))
        if not self.audio_ready:
            canvas.create_text(
                left,
                88,
                text="Audio device is not available. Check Windows sound output and restart the app.",
                anchor="w",
                fill="#b91c1c",
                font=("Segoe UI", 10, "bold"),
            )

        canvas.create_rectangle(left, bar_y, right, bar_y + bar_height, fill=axis, outline="")

        for start, end, evidence_type, label in hit_ranges:
            x1 = left + (start / duration) * usable_width
            x2 = left + (end / duration) * usable_width
            color = self.timeline_color(evidence_type)
            canvas.create_rectangle(x1, bar_y - 8, max(x2, x1 + 3), bar_y + bar_height + 8, fill=color, outline="")
            canvas.create_text(
                (x1 + max(x2, x1 + 3)) / 2,
                bar_y - 18,
                text=label,
                fill=color,
                font=("Segoe UI", 9, "bold"),
            )

        for fraction in (0, 0.25, 0.5, 0.75, 1):
            x = left + fraction * usable_width
            seconds = duration * fraction
            canvas.create_line(x, bar_y + bar_height + 6, x, bar_y + bar_height + 14, fill=axis)
            canvas.create_text(
                x,
                bar_y + bar_height + 30,
                text=self.format_audio_time(seconds),
                fill=muted,
                font=("Segoe UI", 9),
            )

        legend = [("Exact", "#b91c1c"), ("Synonym", "#1d4ed8"), ("Context", "#eab308")]
        legend_y = height - 42
        legend_x = left
        for label, color in legend:
            canvas.create_rectangle(legend_x, legend_y, legend_x + 16, legend_y + 16, fill=color, outline="")
            canvas.create_text(legend_x + 22, legend_y + 8, text=label, anchor="w", fill=text_color, font=("Segoe UI", 10))
            legend_x += 110

        position = self.current_audio_position()
        if position > 0 or self.is_audio_playing or self.is_audio_paused:
            playhead_x = left + min(position / duration, 1.0) * usable_width
            playhead_color = self.theme_value("text")
            canvas.create_line(playhead_x, bar_y - 22, playhead_x, bar_y + bar_height + 22, fill=playhead_color, width=2)
            canvas.create_oval(playhead_x - 5, bar_y - 27, playhead_x + 5, bar_y - 17, fill=playhead_color, outline="")

        if hasattr(self, "audio_time_label"):
            self.audio_time_label.configure(
                text=f"{self.format_audio_time(position)} / {self.format_audio_time(duration)}"
            )

    def transcript_timestamp_ranges(self, transcript_text: str) -> list[tuple[int, float, float]]:
        ranges: list[tuple[int, float, float]] = []
        for line_number, line in enumerate(transcript_text.splitlines(), start=1):
            match = TIMESTAMP_RE.search(line)
            if not match:
                continue
            start = self.parse_audio_time(match.group(1))
            end = self.parse_audio_time(match.group(2))
            ranges.append((line_number, start, max(end, start + 0.2)))
        return ranges

    def hit_timestamp_ranges(self, ranges: list[tuple[int, float, float]]) -> list[tuple[float, float, str, str]]:
        ranges_by_line = {line_number: (start, end) for line_number, start, end in ranges}
        hit_ranges: list[tuple[float, float, str, str]] = []

        for hit in self.hits:
            range_pair = ranges_by_line.get(hit.line_number)
            if not range_pair:
                range_pair = self.timestamp_from_text(hit.text)
            if not range_pair:
                continue
            start, end = range_pair
            label = TYPE_STYLES.get(hit.evidence_type, TYPE_STYLES["exact"])["label"]
            hit_ranges.append((start, end, hit.evidence_type, label))

        return hit_ranges

    def timestamp_from_text(self, text: str) -> tuple[float, float] | None:
        match = TIMESTAMP_RE.search(text)
        if not match:
            return None
        start = self.parse_audio_time(match.group(1))
        end = self.parse_audio_time(match.group(2))
        return start, max(end, start + 0.2)

    def parse_audio_time(self, value: str) -> float:
        parts = value.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)

    def format_audio_time(self, seconds: float) -> str:
        total_seconds = int(seconds)
        minutes = total_seconds // 60
        seconds_part = total_seconds % 60
        return f"{minutes:02d}:{seconds_part:02d}"

    def timeline_color(self, evidence_type: str) -> str:
        return {
            "exact": "#b91c1c",
            "synonym": "#1d4ed8",
            "context": "#eab308",
        }.get(evidence_type, "#64748b")

    def current_audio_path(self) -> Path | None:
        if self.selected_audio_files:
            return self.selected_audio_files[0]

        audio_path = Path(self.audio_dir.get())
        if audio_path.exists() and audio_path.is_file():
            return audio_path
        if audio_path.exists() and audio_path.is_dir():
            audio_files = list_audio_files(audio_path)
            return audio_files[0] if audio_files else None
        return None

    def current_audio_duration(self) -> float:
        audio_path = self.current_audio_path()
        if audio_path is None or audio_path.suffix.casefold() != ".wav":
            return 0.0

        try:
            with wave.open(str(audio_path), "rb") as audio_file:
                frames = audio_file.getnframes()
                rate = audio_file.getframerate()
                return frames / float(rate)
        except (wave.Error, OSError, ZeroDivisionError):
            return 0.0

    def play_audio(self) -> None:
        if not self.audio_ready:
            messagebox.showerror("Audio unavailable", "Audio playback is not available on this system.")
            return

        audio_path = self.current_audio_path()
        if audio_path is None:
            messagebox.showwarning("No audio", "Choose or transcribe an audio file before playing.")
            return

        try:
            if self.is_audio_paused and self.playing_audio == audio_path and self.audio_channel:
                self.audio_channel.unpause()
                self.play_started_at = time.monotonic()
                self.is_audio_paused = False
                self.is_audio_playing = True
            else:
                self.playing_audio = audio_path
                self.audio_duration = self.current_audio_duration() or self.audio_duration
                self.audio_sound = self.load_audio_sound(audio_path, self.play_start_offset)
                self.audio_channel = self.audio_sound.play()
                if self.audio_channel is None:
                    raise pygame.error("No audio channel is available for playback.")
                self.play_started_at = time.monotonic()
                self.is_audio_playing = True
                self.is_audio_paused = False
                self.set_audio_status(f"Playing {audio_path.name}")
        except Exception as exc:
            messagebox.showerror("Playback failed", str(exc))
            self.set_audio_status("Playback failed")
            return

        self.schedule_playhead_update()

    def pause_audio(self) -> None:
        if not self.is_audio_playing:
            return

        self.play_start_offset = self.current_audio_position()
        if self.audio_channel:
            self.audio_channel.pause()
        self.is_audio_playing = False
        self.is_audio_paused = True
        self.set_audio_status("Paused")
        self.render_audio_timeline()

    def stop_audio(self) -> None:
        if self.playhead_after_id:
            self.after_cancel(self.playhead_after_id)
            self.playhead_after_id = None
        if self.audio_channel:
            self.audio_channel.stop()
            self.audio_channel = None
        self.play_start_offset = 0.0
        self.play_started_at = 0.0
        self.is_audio_playing = False
        self.is_audio_paused = False
        self.set_audio_status("")
        self.render_audio_timeline()

    def reset_audio_playback(self) -> None:
        if hasattr(self, "audio_timeline"):
            self.stop_audio()

    def seek_audio_from_timeline(self, event) -> None:
        if not hasattr(self, "timeline_geometry"):
            return

        left, right, _bar_y, _bar_height, duration = self.timeline_geometry
        if event.x < left or event.x > right:
            return

        fraction = (event.x - left) / max(right - left, 1)
        self.play_start_offset = max(0.0, min(duration * fraction, duration))
        was_playing = self.is_audio_playing
        if was_playing:
            if self.audio_channel:
                self.audio_channel.stop()
            self.is_audio_playing = False
            self.is_audio_paused = False
            self.play_audio()
        else:
            self.is_audio_paused = True
            self.render_audio_timeline()

    def current_audio_position(self) -> float:
        if self.is_audio_playing:
            elapsed = max(0.0, time.monotonic() - self.play_started_at)
            return min(self.play_start_offset + elapsed, self.audio_duration or self.play_start_offset + elapsed)
        if self.is_audio_paused:
            return self.play_start_offset
        return 0.0

    def schedule_playhead_update(self) -> None:
        if self.playhead_after_id:
            self.after_cancel(self.playhead_after_id)
        self.playhead_after_id = self.after(250, self.update_playhead)

    def update_playhead(self) -> None:
        self.render_audio_timeline()
        if self.is_audio_playing and self.audio_channel and self.audio_channel.get_busy():
            self.schedule_playhead_update()
        elif self.is_audio_playing:
            self.stop_audio()

    def load_audio_sound(self, audio_path: Path, start_seconds: float = 0.0):
        if audio_path.suffix.casefold() != ".wav" or start_seconds <= 0:
            return pygame.mixer.Sound(str(audio_path))

        with wave.open(str(audio_path), "rb") as audio_file:
            channels = audio_file.getnchannels()
            sample_width = audio_file.getsampwidth()
            sample_rate = audio_file.getframerate()
            start_frame = int(start_seconds * sample_rate)
            audio_file.setpos(min(start_frame, audio_file.getnframes()))
            frames = audio_file.readframes(audio_file.getnframes() - start_frame)

        pygame.mixer.quit()
        pygame.mixer.init(frequency=sample_rate, size=-8 * sample_width, channels=channels)
        return pygame.mixer.Sound(buffer=frames)

    def set_audio_status(self, status: str) -> None:
        if hasattr(self, "audio_playback_status"):
            self.audio_playback_status.configure(text=status)

    def current_audio_names(self) -> list[str]:
        if self.selected_audio_files:
            return [path.name for path in self.selected_audio_files]

        audio_path = Path(self.audio_dir.get())
        if audio_path.exists() and audio_path.is_file():
            return [audio_path.name]
        if audio_path.exists() and audio_path.is_dir():
            return [path.name for path in list_audio_files(audio_path)]
        return []

    def render_graph(self) -> None:
        if not hasattr(self, "graph_canvas"):
            return

        canvas = self.graph_canvas
        canvas.configure(bg=self.theme_value("graph_bg"), highlightbackground=self.theme_value("border"))
        canvas.delete("all")
        width = max(canvas.winfo_width(), 400)
        height = max(canvas.winfo_height(), 280)

        counts = [
            ("Exact", self.count_hits("exact"), "#b91c1c"),
            ("Synonym", self.count_hits("synonym"), "#1d4ed8"),
            ("Context", self.count_hits("context"), "#eab308"),
        ]
        max_count = max([count for _label, count, _color in counts] + [1])

        left = 72
        right = width - 38
        top = 48
        bottom = height - 72
        chart_width = max(right - left, 100)
        chart_height = max(bottom - top, 100)

        canvas.create_text(
            left,
            24,
            text="Finding Types",
            anchor="w",
            fill=self.theme_value("text"),
            font=("Segoe UI", 16, "bold"),
        )
        canvas.create_line(left, bottom, right, bottom, fill=self.theme_value("axis"), width=2)
        canvas.create_line(left, top, left, bottom, fill=self.theme_value("axis"), width=2)

        bar_area = chart_width / len(counts)
        bar_width = min(96, bar_area * 0.48)

        for index, (label, count, color) in enumerate(counts):
            center_x = left + bar_area * index + bar_area / 2
            bar_height = (count / max_count) * chart_height if count else 0
            x1 = center_x - bar_width / 2
            x2 = center_x + bar_width / 2
            y1 = bottom - bar_height

            canvas.create_rectangle(x1, y1, x2, bottom, fill=color, outline="")
            canvas.create_text(center_x, y1 - 14, text=str(count), fill=self.theme_value("text"), font=("Segoe UI", 12, "bold"))
            canvas.create_text(center_x, bottom + 24, text=label, fill=self.theme_value("muted"), font=("Segoe UI", 11))

        canvas.create_text(
            left,
            height - 26,
            text=f"Total findings: {len(self.hits)}",
            anchor="w",
            fill=self.theme_value("muted"),
            font=("Segoe UI", 11),
        )

    def count_hits(self, evidence_type: str) -> int:
        return sum(1 for hit in self.hits if hit.evidence_type == evidence_type)

    def export_results(self) -> None:
        if not self.hits:
            messagebox.showinfo("No results", "Run a scan before exporting.")
            return

        path = filedialog.asksaveasfilename(
            title="Export results JSON",
            defaultextension=".json",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            initialfile="transcript_hits.json",
        )
        if not path:
            return

        Path(path).write_text(
            json.dumps([asdict(hit) for hit in self.hits], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.status.set(f"Exported {len(self.hits)} hits")

    def refresh_outputs(self) -> None:
        self.stop_audio()
        self.hits = []
        self.generated_transcript_paths = []
        self.current_transcript_text = ""

        self.transcribed_text.configure(state="normal")
        self.transcribed_text.delete("1.0", tk.END)

        self.results.configure(state="normal")
        self.results.delete("1.0", tk.END)
        self.results.configure(state="disabled")

        self.total_label.configure(text="0")
        self.exact_label.configure(text="0")
        self.synonym_label.configure(text="0")
        self.context_label.configure(text="0")

        if hasattr(self, "audio_timeline"):
            self.audio_timeline.delete("all")
        if hasattr(self, "graph_canvas"):
            self.graph_canvas.delete("all")
        self.status.set("Ready")


def main() -> None:
    app = TranscriptScannerApp()
    app.mainloop()


if __name__ == "__main__":
    main()

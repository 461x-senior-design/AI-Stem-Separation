#!/usr/bin/env python3
"""PySide6 front end for Stemmy source separation."""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPalette, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from stemmy.constants import STEMS_4
from stemmy.inference import (
    InferenceConfig,
    config_from_checkpoint,
    load_pth_model,
    separate_audio_file,
)
from stemmy.logging_config import setup_logging

TITLE_FONT_SIZE = 80
UI_FONT_FILE = "Inter-VariableFont_opsz,wght.ttf"
UI_FONT_FAMILY_FALLBACK = "Inter"
DARK_ACCENT_COLOR = "#7FE80E"
LIGHT_ACCENT_COLOR = "#2F7D1C"
SOLO_ACCENT_COLOR = "#FF6A00"
DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parents[3] / "checkpoints" / "model.pth"
TRACE_COLORS = {
    "drums": "#00C2FF",
    "bass": "#FF2E88",
    "vocals": "#FFE600",
    "other": "#00FF85",
}
OSCILLOSCOPE_POINTS = 2400


@dataclass(frozen=True)
class SeparationJob:
    """User-selected separation options."""

    input_file: Path
    output_dir: Path
    checkpoint: Optional[Path]
    device: str
    chunk_frames: int
    overlap_frames: int
    amp: bool


class SeparationWorker(QObject):
    """Run model loading and separation off the UI thread."""

    log = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, job: SeparationJob) -> None:
        super().__init__()
        self.job = job

    @Slot()
    def run(self) -> None:
        """Execute the separation job."""
        try:
            stems = list(STEMS_4)
            self.log.emit("Loading model...")

            model, cfg, checkpoint_obj = self._load_model(stems)
            cfg = replace(
                cfg,
                device=self.job.device,
                stems=stems,
                export_files=True,
                renorm_masks=True,
                chunk_frames=self.job.chunk_frames,
                overlap_frames=self.job.overlap_frames,
                amp=self.job.amp,
            )

            self.log.emit("Separating audio...")
            outputs = separate_audio_file(
                audio_path=self.job.input_file,
                model=model,
                cfg=cfg,
                output_dir=self.job.output_dir,
                export_files=True,
                stems=stems,
                checkpoint=checkpoint_obj,
            )

            paths = {stem: str(path) for stem, path in outputs.paths.items()}
            self.finished.emit(paths)
        except Exception as exc:  # noqa: BLE001 - signal the full UI-facing failure.
            details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.failed.emit(details)

    def _load_model(self, stems: list[str]) -> tuple[torch.nn.Module, InferenceConfig, object]:
        if self.job.checkpoint is not None:
            model, checkpoint_obj = load_pth_model(
                self.job.checkpoint,
                device=self.job.device,
                stems=len(stems),
            )
            return model, config_from_checkpoint(checkpoint_obj), checkpoint_obj

        raise ValueError("Default checkpoint is not configured.")


class OscilloscopeWidget(QWidget):
    """Draw downsampled per-stem waveform traces."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Oscilloscope")
        self.setMinimumHeight(170)
        self.traces: dict[str, np.ndarray] = {}
        self.levels: dict[str, float] = {}

    def clear(self) -> None:
        self.traces.clear()
        self.levels.clear()
        self.update()

    def set_traces(self, traces: dict[str, np.ndarray]) -> None:
        self.traces = traces
        self.levels = {stem: 1.0 for stem in traces}
        self.update()

    def set_levels(self, levels: dict[str, float]) -> None:
        self.levels = levels
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override name.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.palette().color(QPalette.ColorRole.Base))

        rect = self.rect().adjusted(10, 10, -10, -10)
        painter.setPen(QPen(QColor("#6b7280"), 1))
        painter.drawLine(rect.left(), rect.center().y(), rect.right(), rect.center().y())

        if not self.traces:
            painter.setPen(QColor("#9ca3af"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Oscilloscope")
            return

        painter.setClipRect(rect)
        center_y = rect.center().y()
        half_height = max(1, rect.height() / 2 - 6)

        legend_x = rect.left() + 6
        for stem in STEMS_4:
            trace = self.traces.get(stem)
            if trace is None or len(trace) < 2:
                continue

            level = self.levels.get(stem, 1.0)
            color = QColor(TRACE_COLORS.get(stem, "#e5e7eb"))
            legend_color = QColor(color)
            legend_color.setAlphaF(0.95)
            painter.setPen(QPen(legend_color, 2.0))
            painter.drawText(legend_x, rect.top() + 14, stem.title())
            legend_x += 64

            if level <= 0:
                continue

            color.setAlphaF(0.25)
            painter.setPen(QPen(color, 2.4))

            points = len(trace)
            prev_x = rect.left()
            prev_y = int(center_y - float(trace[0]) * level * half_height)
            for idx in range(1, points):
                x = rect.left() + int(idx * rect.width() / (points - 1))
                y = int(center_y - float(trace[idx]) * level * half_height)
                painter.drawLine(prev_x, prev_y, x, y)
                prev_x = x
                prev_y = y


class MainWindow(QMainWindow):
    """Stemmy desktop front end."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Stemmy")
        self.setMinimumSize(860, 640)
        ui_font_family = _load_font_family(Path(__file__).with_name(UI_FONT_FILE))
        self.setFont(QFont(ui_font_family or UI_FONT_FAMILY_FALLBACK))
        self.title_font_family = _load_font_family(Path(__file__).with_name("adrip1.ttf"))

        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[SeparationWorker] = None
        self.stem_paths: dict[str, Path] = {}
        self.audio_outputs: dict[str, QAudioOutput] = {}
        self.players: dict[str, QMediaPlayer] = {}
        self.volume_sliders: dict[str, QSlider] = {}
        self.solo_buttons: dict[str, QPushButton] = {}

        for stem in STEMS_4:
            audio_output = QAudioOutput(self)
            audio_output.setVolume(0.85)
            player = QMediaPlayer(self)
            player.setAudioOutput(audio_output)
            player.errorOccurred.connect(
                lambda _error, error_text, stem=stem: self._on_playback_error(stem, error_text)
            )
            player.playbackStateChanged.connect(self._on_playback_state_changed)
            self.audio_outputs[stem] = audio_output
            self.players[stem] = player

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Choose an audio file")

        self.output_edit = QLineEdit(str((Path.cwd() / "separated").resolve()))

        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU", "cpu")
        if torch.cuda.is_available():
            self.device_combo.addItem("CUDA", "cuda")
            for idx in range(torch.cuda.device_count()):
                self.device_combo.addItem(f"CUDA:{idx}", f"cuda:{idx}")

        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(0, 100_000)
        self.chunk_spin.setValue(256)
        self.chunk_spin.setSuffix(" frames")

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 100_000)
        self.overlap_spin.setValue(64)
        self.overlap_spin.setSuffix(" frames")

        self.amp_check = QCheckBox("AMP")
        self.amp_check.setEnabled(torch.cuda.is_available())

        self.dark_mode_check = QCheckBox("Dark Mode")
        self.dark_mode_check.setChecked(True)
        self.dark_mode_check.toggled.connect(self._apply_style)

        self.run_button = QPushButton("Run Separation")
        self.run_button.clicked.connect(self.start_separation)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

        self.outputs_list = QListWidget()
        self.outputs_list.currentItemChanged.connect(self._on_output_selection_changed)
        self.oscilloscope = OscilloscopeWidget()
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle_stem_mix)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_playback)
        self.now_playing_label = QLabel("No stems loaded")
        self.now_playing_label.setObjectName("NowPlaying")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Run details appear here.")

        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        header_row = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("STEMMY")
        title.setObjectName("Title")
        if self.title_font_family is not None:
            title.setFont(QFont(self.title_font_family, TITLE_FONT_SIZE, QFont.Weight.Bold))
        subtitle = QLabel("Separate an audio file into drums, bass, vocals, and other stems.")
        subtitle.setObjectName("Subtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header_row.addLayout(title_block, 1)
        header_row.addWidget(self.dark_mode_check, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header_row)

        files_group = QGroupBox("Files")
        files_form = QFormLayout(files_group)
        files_form.addRow(
            self._form_label("Input audio"),
            self._path_picker_row(self.input_edit, self._pick_input),
        )
        files_form.addRow(
            self._form_label("Output folder"),
            self._path_picker_row(self.output_edit, self._pick_output_dir),
        )
        layout.addWidget(files_group)

        options_group = QGroupBox("Inference")
        options_layout = QHBoxLayout(options_group)
        options_layout.addWidget(self._labeled_widget("Device", self.device_combo))
        options_layout.addWidget(self._labeled_widget("Chunk", self.chunk_spin))
        options_layout.addWidget(self._labeled_widget("Overlap", self.overlap_spin))
        options_layout.addWidget(self.amp_check)
        options_layout.addStretch(1)
        layout.addWidget(options_group)

        action_row = QHBoxLayout()
        action_row.addWidget(self.progress, 1)
        action_row.addWidget(self.run_button)
        layout.addLayout(action_row)

        details = QVBoxLayout()
        mixer_panel = QWidget()
        mixer_layout = QVBoxLayout(mixer_panel)
        mixer_layout.setContentsMargins(0, 0, 0, 0)
        mixer_layout.setSpacing(8)
        mixer_layout.addWidget(self.oscilloscope)
        mixer_layout.addWidget(self._mixer_widget())

        playback_row = QHBoxLayout()
        playback_row.addWidget(self.play_button)
        playback_row.addWidget(self.stop_button)
        playback_row.addWidget(self.now_playing_label, 1)
        mixer_layout.addLayout(playback_row)

        log_frame = self._details_frame("Log", self.log_view)
        mixer_frame = self._details_frame("Mixer", mixer_panel)
        outputs_frame = self._details_frame("Output Stems", self.outputs_list)
        details.addWidget(log_frame, 1)
        details.addWidget(mixer_frame, 2)
        details.addWidget(outputs_frame, 1)
        layout.addLayout(details, 1)

    @Slot()
    @Slot(bool)
    def _apply_style(self, _checked: bool = False) -> None:
        if self.dark_mode_check.isChecked():
            self._apply_window_palette(dark=True)
            self.setStyleSheet(
                """
            QMainWindow, QWidget {
                background: #111827;
                color: #e5e7eb;
                font-size: 14px;
            }
            QLabel#Title {
                font-size: %dpx;
                font-weight: 700;
                color: %s;
            }
            QLabel#Subtitle {
                color: #a7b0bd;
                font-size: 15px;
            }
            QGroupBox, QFrame#DetailsFrame {
                background: #1f2937;
                border: 1px solid #374151;
                border-radius: 8px;
                margin-top: 10px;
                padding: 14px;
            }
            QFrame#MixerFrame {
                background: #111827;
                border: 1px solid #4b5563;
                border-radius: 8px;
            }
            QWidget#Oscilloscope {
                background: #0f172a;
                border: 1px solid #4b5563;
                border-radius: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
                color: #d1d5db;
                font-weight: 600;
            }
            QLabel#DetailsHeading {
                color: #d1d5db;
                font-weight: 600;
            }
            QLabel#NowPlaying {
                color: #a7b0bd;
            }
            QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QListWidget {
                background: #0f172a;
                border: 1px solid #4b5563;
                border-radius: 6px;
                color: #f9fafb;
                padding: 6px;
            }
            QLineEdit:disabled {
                background: #1f2937;
                color: #6b7280;
            }
            QPushButton {
                background: %s;
                color: #111827;
                border: 0;
                border-radius: 6px;
                padding: 8px 14px;
                font-weight: 700;
            }
            QPushButton:disabled {
                background: #3f5f2b;
                color: #9ca3af;
            }
            QPushButton:checked {
                background: %s;
                color: #111827;
            }
            QSlider::groove:vertical {
                background: #d1d5db;
                width: 2px;
                border-radius: 1px;
            }
            QSlider::handle:vertical {
                background: #14532d;
                border: 1px solid #d1d5db;
                height: 16px;
                margin: 0 -10px;
                border-radius: 4px;
            }
            QProgressBar {
                border: 1px solid #4b5563;
                border-radius: 6px;
                height: 18px;
                text-align: center;
                background: #0f172a;
                color: #f9fafb;
            }
            QProgressBar::chunk {
                background: %s;
                border-radius: 5px;
            }
            """
                % (
                    TITLE_FONT_SIZE,
                    DARK_ACCENT_COLOR,
                    DARK_ACCENT_COLOR,
                    SOLO_ACCENT_COLOR,
                    DARK_ACCENT_COLOR,
                )
            )
            return

        self._apply_window_palette(dark=False)
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f5f6f8;
                color: #1f2933;
                font-size: 14px;
            }
            QLabel#Title {
                font-size: %dpx;
                font-weight: 700;
                color: %s;
            }
            QLabel#Subtitle {
                color: #52606d;
                font-size: 15px;
            }
            QGroupBox, QFrame#DetailsFrame {
                background: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 8px;
                margin-top: 10px;
                padding: 14px;
            }
            QFrame#MixerFrame {
                background: #f5f6f8;
                border: 1px solid #bcccdc;
                border-radius: 8px;
            }
            QWidget#Oscilloscope {
                background: #ffffff;
                border: 1px solid #bcccdc;
                border-radius: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
                color: #334e68;
                font-weight: 600;
            }
            QLabel#DetailsHeading {
                color: #334e68;
                font-weight: 600;
            }
            QLabel#NowPlaying {
                color: #52606d;
            }
            QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QListWidget {
                background: #ffffff;
                border: 1px solid #bcccdc;
                border-radius: 6px;
                padding: 6px;
            }
            QPushButton {
                background: %s;
                color: #ffffff;
                border: 0;
                border-radius: 6px;
                padding: 8px 14px;
                font-weight: 700;
            }
            QPushButton:disabled {
                background: #b7d69a;
                color: #52606d;
            }
            QPushButton:checked {
                background: %s;
                color: #111827;
            }
            QSlider::groove:vertical {
                background: #9aa6b2;
                width: 2px;
                border-radius: 1px;
            }
            QSlider::handle:vertical {
                background: #ffffff;
                border: 1px solid #52606d;
                height: 16px;
                margin: 0 -10px;
                border-radius: 4px;
            }
            QProgressBar {
                border: 1px solid #bcccdc;
                border-radius: 6px;
                height: 18px;
                text-align: center;
                background: #ffffff;
            }
            QProgressBar::chunk {
                background: %s;
                border-radius: 5px;
            }
            """
            % (
                TITLE_FONT_SIZE,
                LIGHT_ACCENT_COLOR,
                LIGHT_ACCENT_COLOR,
                SOLO_ACCENT_COLOR,
                LIGHT_ACCENT_COLOR,
            )
        )

    def _apply_window_palette(self, dark: bool) -> None:
        palette = QPalette()
        if dark:
            palette.setColor(QPalette.ColorRole.Window, QColor("#111827"))
            palette.setColor(QPalette.ColorRole.WindowText, QColor("#e5e7eb"))
            palette.setColor(QPalette.ColorRole.Base, QColor("#0f172a"))
            palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#1f2937"))
            palette.setColor(QPalette.ColorRole.Text, QColor("#f9fafb"))
            palette.setColor(QPalette.ColorRole.Button, QColor(DARK_ACCENT_COLOR))
            palette.setColor(QPalette.ColorRole.ButtonText, QColor("#111827"))
            palette.setColor(QPalette.ColorRole.Highlight, QColor(DARK_ACCENT_COLOR))
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#111827"))
        else:
            palette.setColor(QPalette.ColorRole.Window, QColor("#f5f6f8"))
            palette.setColor(QPalette.ColorRole.WindowText, QColor("#1f2933"))
            palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
            palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f5f6f8"))
            palette.setColor(QPalette.ColorRole.Text, QColor("#1f2933"))
            palette.setColor(QPalette.ColorRole.Button, QColor(LIGHT_ACCENT_COLOR))
            palette.setColor(QPalette.ColorRole.ButtonText, QColor("#ffffff"))
            palette.setColor(QPalette.ColorRole.Highlight, QColor(LIGHT_ACCENT_COLOR))
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))

        app = QApplication.instance()
        if app is not None:
            app.setPalette(palette)
        self.setPalette(palette)

    def _path_picker_row(self, edit: QLineEdit, slot) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        browse = QPushButton("Browse")
        browse.clicked.connect(slot)
        row_layout.addWidget(edit, 1)
        row_layout.addWidget(browse)
        return row

    def _labeled_widget(self, label_text: str, widget: QWidget) -> QWidget:
        box = QWidget()
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(label_text)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        box_layout.addWidget(label)
        box_layout.addWidget(widget)
        return box

    def _form_label(self, label_text: str) -> QLabel:
        label = QLabel(label_text)
        label.setContentsMargins(6, 0, 6, 0)
        return label

    def _mixer_widget(self) -> QWidget:
        mixer = QFrame()
        mixer.setObjectName("MixerFrame")
        mixer_layout = QHBoxLayout(mixer)
        mixer_layout.setContentsMargins(12, 12, 12, 12)
        mixer_layout.setSpacing(18)

        for stem in STEMS_4:
            strip = QWidget()
            strip_layout = QVBoxLayout(strip)
            strip_layout.setContentsMargins(0, 0, 0, 0)
            strip_layout.setSpacing(6)

            stem_label = QLabel(stem.title())
            stem_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

            slider = QSlider(Qt.Orientation.Vertical)
            slider.setRange(0, 100)
            slider.setValue(85)
            slider.setFixedHeight(140)
            slider.setToolTip(f"{stem.title()} volume")
            slider.valueChanged.connect(lambda _value, stem=stem: self._apply_mixer_volumes())
            self.volume_sliders[stem] = slider

            solo_button = QPushButton("S")
            solo_button.setCheckable(True)
            solo_button.setFixedWidth(36)
            solo_button.setToolTip(f"Solo {stem}")
            solo_button.toggled.connect(
                lambda checked, stem=stem: self._on_solo_toggled(stem, checked)
            )
            self.solo_buttons[stem] = solo_button

            strip_layout.addWidget(stem_label)
            strip_layout.addWidget(slider, 1, Qt.AlignmentFlag.AlignHCenter)
            strip_layout.addWidget(solo_button, 0, Qt.AlignmentFlag.AlignHCenter)
            mixer_layout.addWidget(strip)

        return mixer

    def _details_frame(self, title: str, widget: QWidget) -> QFrame:
        frame = QFrame()
        frame.setObjectName("DetailsFrame")
        frame_layout = QVBoxLayout(frame)
        heading = QLabel(title)
        heading.setObjectName("DetailsHeading")
        frame_layout.addWidget(heading)
        frame_layout.addWidget(widget, 1)
        return frame

    @Slot()
    def _pick_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose audio file",
            str(Path.cwd()),
            "Audio Files (*.wav *.flac *.mp3 *.ogg *.aiff *.aif);;All Files (*)",
        )
        if path:
            self.input_edit.setText(path)

    @Slot()
    def _pick_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output folder", str(Path.cwd()))
        if path:
            self.output_edit.setText(path)

    @Slot()
    def start_separation(self) -> None:
        job = self._build_job()
        if job is None:
            return

        self.outputs_list.clear()
        self.stop_playback()
        self.stem_paths.clear()
        for player in self.players.values():
            player.setSource(QUrl())
        self.oscilloscope.clear()
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.now_playing_label.setText("No stems loaded")
        self.log_view.clear()
        self._set_running(True)
        self._append_log(f"Input: {job.input_file}")
        self._append_log(f"Output: {job.output_dir}")
        self._append_log(f"Checkpoint: {job.checkpoint}")
        self._append_log(f"Device: {job.device}")

        self.worker_thread = QThread(self)
        self.worker = SeparationWorker(job)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log.connect(self._append_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._clear_worker_refs)
        self.worker_thread.start()

    def _build_job(self) -> Optional[SeparationJob]:
        try:
            input_file = _validated_file(self.input_edit.text(), "Input audio")
            output_dir = _validated_output_dir(self.output_edit.text())
            checkpoint = _validated_file(str(DEFAULT_CHECKPOINT_PATH), "Default checkpoint")

            chunk_frames = int(self.chunk_spin.value())
            overlap_frames = int(self.overlap_spin.value())
            if chunk_frames > 0 and overlap_frames >= chunk_frames:
                raise ValueError("Overlap frames must be less than chunk frames.")

            device = str(self.device_combo.currentData())
            if self.amp_check.isChecked() and not device.startswith("cuda"):
                raise ValueError("AMP requires a CUDA device.")

            return SeparationJob(
                input_file=input_file,
                output_dir=output_dir,
                checkpoint=checkpoint,
                device=device,
                chunk_frames=chunk_frames,
                overlap_frames=overlap_frames,
                amp=self.amp_check.isChecked(),
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid Settings", str(exc))
            return None

    def _set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.progress.setRange(0, 0 if running else 1)
        self.progress.setValue(0)

    @Slot(str)
    def _append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    @Slot(dict)
    def _on_finished(self, paths: dict) -> None:
        self._set_running(False)
        self._append_log("Done.")
        loaded_count = 0
        for stem in list(STEMS_4):
            path = paths.get(stem)
            if path:
                resolved_path = Path(str(path)).expanduser().resolve()
                self.stem_paths[stem] = resolved_path
                self.players[stem].setSource(QUrl.fromLocalFile(str(resolved_path)))
                item = QListWidgetItem(f"{stem}: {path}")
                item.setData(Qt.ItemDataRole.UserRole, path)
                self.outputs_list.addItem(item)
                loaded_count += 1
        if self.outputs_list.count() > 0:
            self.outputs_list.setCurrentRow(0)
        self.play_button.setEnabled(loaded_count > 0)
        self._load_oscilloscope_traces()
        self._apply_mixer_volumes()
        self.now_playing_label.setText(f"Ready to play {loaded_count} stems")
        QMessageBox.information(self, "Separation Complete", "Stem separation finished.")

    @Slot(str)
    def _on_failed(self, details: str) -> None:
        self._set_running(False)
        self._append_log(details)
        QMessageBox.critical(self, "Separation Failed", _last_error_line(details))

    @Slot()
    def _clear_worker_refs(self) -> None:
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
        self.worker_thread = None
        self.worker = None

    @Slot()
    def toggle_stem_mix(self) -> None:
        if not self.stem_paths:
            QMessageBox.information(self, "No Stems Loaded", "Run separation before playing stems.")
            return

        if self._any_player_in_state(QMediaPlayer.PlaybackState.PlayingState):
            for player in self.players.values():
                player.pause()
            return

        restart = not self._any_player_in_state(QMediaPlayer.PlaybackState.PausedState)
        for stem, player in self.players.items():
            if stem not in self.stem_paths:
                continue
            if restart:
                player.setPosition(0)
            player.play()
        self.now_playing_label.setText("Playing stem mix")

    def _any_player_in_state(self, state: QMediaPlayer.PlaybackState) -> bool:
        return any(player.playbackState() == state for player in self.players.values())

    @Slot()
    def stop_playback(self) -> None:
        for player in self.players.values():
            player.stop()

    @Slot(QListWidgetItem, QListWidgetItem)
    def _on_output_selection_changed(
        self,
        current: Optional[QListWidgetItem],
        _previous: Optional[QListWidgetItem],
    ) -> None:
        self.play_button.setEnabled(bool(self.stem_paths))
        players_stopped = not self._any_player_in_state(QMediaPlayer.PlaybackState.PlayingState)
        if current is not None and players_stopped:
            path = _item_path(current)
            if path is not None:
                self.now_playing_label.setText(path.name)

    @Slot(QMediaPlayer.PlaybackState)
    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        is_playing = self._any_player_in_state(QMediaPlayer.PlaybackState.PlayingState)
        self.play_button.setText("Pause" if is_playing else "Play")
        any_active = is_playing or self._any_player_in_state(QMediaPlayer.PlaybackState.PausedState)
        self.stop_button.setEnabled(any_active)
        if state == QMediaPlayer.PlaybackState.StoppedState and not any_active and self.stem_paths:
            self.now_playing_label.setText("Stem mix stopped")

    def _apply_mixer_volumes(self) -> None:
        soloed_stems = {
            stem for stem, button in self.solo_buttons.items() if button.isChecked()
        }
        levels: dict[str, float] = {}
        for stem, audio_output in self.audio_outputs.items():
            slider = self.volume_sliders.get(stem)
            if slider is None:
                continue
            volume = slider.value() / 100.0
            if soloed_stems and stem not in soloed_stems:
                volume = 0.0
            audio_output.setVolume(volume)
            levels[stem] = volume
        self.oscilloscope.set_levels(levels)

    def _on_solo_toggled(self, solo_stem: str, checked: bool) -> None:
        if checked:
            for stem, button in self.solo_buttons.items():
                if stem == solo_stem or not button.isChecked():
                    continue
                button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(False)
        self._apply_mixer_volumes()

    def _load_oscilloscope_traces(self) -> None:
        traces: dict[str, np.ndarray] = {}
        for stem in STEMS_4:
            path = self.stem_paths.get(stem)
            if path is None:
                continue
            try:
                traces[stem] = _load_waveform_trace(path)
            except (OSError, RuntimeError, ValueError) as exc:
                self._append_log(f"Could not load oscilloscope trace for {stem}: {exc}")
        self.oscilloscope.set_traces(traces)

    def _on_playback_error(self, stem: str, error_text: str) -> None:
        if error_text:
            QMessageBox.warning(self, "Playback Failed", f"{stem}: {error_text}")


def _validated_file(raw_path: str, label: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not raw_path.strip():
        raise ValueError(f"{label} path is required.")
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"{label} must be a file: {path}")
    return path


def _validated_output_dir(raw_path: str) -> Path:
    if not raw_path.strip():
        raise ValueError("Output folder is required.")
    path = Path(raw_path).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Output path must be a folder: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_waveform_trace(path: Path) -> np.ndarray:
    audio, _sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.size == 0:
        raise ValueError("empty audio file")

    mono = np.mean(audio, axis=1)
    if len(mono) > OSCILLOSCOPE_POINTS:
        indices = np.linspace(0, len(mono) - 1, OSCILLOSCOPE_POINTS).astype(np.int64)
        mono = mono[indices]

    peak = float(np.max(np.abs(mono)))
    if peak > 0:
        mono = mono / peak
    return mono.astype(np.float32, copy=False)


def _item_path(item: QListWidgetItem) -> Optional[Path]:
    raw_path = item.data(Qt.ItemDataRole.UserRole)
    if raw_path is None:
        return None
    return Path(str(raw_path)).expanduser().resolve()


def _load_font_family(path: Path) -> Optional[str]:
    font_id = QFontDatabase.addApplicationFont(str(path))
    if font_id < 0:
        return None

    families = QFontDatabase.applicationFontFamilies(font_id)
    return families[0] if families else None


def _last_error_line(details: str) -> str:
    lines = [line.strip() for line in details.splitlines() if line.strip()]
    return lines[-1] if lines else "Unknown error."


def main() -> int:
    """Launch the Stemmy GUI."""
    setup_logging(level="ERROR")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

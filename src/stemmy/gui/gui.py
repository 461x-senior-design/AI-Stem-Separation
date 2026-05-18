#!/usr/bin/env python3
"""PySide6 front end for Stemmy source separation."""

from __future__ import annotations

import sys
import threading
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError):  # pragma: no cover - depends on optional native audio deps.
    sd = None
import soundfile as sf
import torch
from PySide6.QtCore import QObject, QPoint, QSettings, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPolygon,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QDialog,
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
    QStyle,
    QStyleOptionSpinBox,
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
SOLO_ACCENT_COLOR = "#FF6A00"
APP_DARK_BG = "#202124"
PANEL_DARK_BG = "#2A2B2F"
FIELD_DARK_BG = "#191A1D"
DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parents[3] / "checkpoints" / "model.pth"
SETTINGS_ORGANIZATION = "Stemmy"
SETTINGS_APPLICATION = "Stemmy"
THEME_SETTINGS_GROUP = "theme"
TRACE_SETTINGS_GROUP = "traceColors"
DEFAULT_THEME_COLORS = {
    "app_bg": APP_DARK_BG,
    "panel_bg": PANEL_DARK_BG,
    "field_bg": FIELD_DARK_BG,
    "text": "#e5e7eb",
    "muted_text": "#a7b0bd",
    "border": "#4b5563",
    "accent": DARK_ACCENT_COLOR,
    "solo": SOLO_ACCENT_COLOR,
}
THEME_COLOR_LABELS = {
    "app_bg": "App",
    "panel_bg": "Panel",
    "field_bg": "Field",
    "text": "Text",
    "muted_text": "Muted",
    "border": "Border",
    "accent": "Accent",
    "solo": "Solo",
}
TRACE_COLORS = {
    "drums": "#00C2FF",
    "bass": "#FF2E88",
    "vocals": "#FFE600",
    "other": "#00FF85",
}
OSCILLOSCOPE_POINTS = 2400
OSCILLOSCOPE_TRACE_WIDTH_DARK = 1.2
OSCILLOSCOPE_TRACE_WIDTH_LIGHT = 1.4
OSCILLOSCOPE_PROGRESS_WIDTH = 2.0
OSCILLOSCOPE_PLAYHEAD_WIDTH = 1.5
MODEL_CACHE_MAX_SIZE = 2
_MODEL_CACHE: dict[tuple[Path, str, int, int, int], tuple[torch.nn.Module, object]] = {}


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
            cache_key = _model_cache_key(self.job.checkpoint, self.job.device, len(stems))
            cached = _MODEL_CACHE.get(cache_key)
            if cached is not None:
                model, checkpoint_obj = cached
                return model, config_from_checkpoint(checkpoint_obj), checkpoint_obj

            model, checkpoint_obj = load_pth_model(
                self.job.checkpoint,
                device=self.job.device,
                stems=len(stems),
            )
            _MODEL_CACHE[cache_key] = (model, checkpoint_obj)
            while len(_MODEL_CACHE) > MODEL_CACHE_MAX_SIZE:
                _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
            return model, config_from_checkpoint(checkpoint_obj), checkpoint_obj

        raise ValueError("Default checkpoint is not configured.")


class StemMixer(QObject):
    """Single-output stem mixer for interactive playback."""

    state_changed = Signal()
    log = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._stems: dict[str, np.ndarray] = {}
        self._sample_rate = 0
        self._frame_count = 0
        self._gains: dict[str, float] = {}
        self._solo_stem: Optional[str] = None
        self._stream = None
        self._cursor_frame = 0
        self._state = "stopped"
        self._lock = threading.RLock()

    def load_stems(self, paths: dict[str, Path]) -> None:
        self.stop()
        stems: dict[str, np.ndarray] = {}
        sample_rate: Optional[int] = None

        for stem in STEMS_4:
            path = paths.get(stem)
            if path is None:
                continue
            audio, read_sample_rate = _read_stem_audio(path)
            if sample_rate is None:
                sample_rate = int(read_sample_rate)
            elif int(read_sample_rate) != sample_rate:
                raise ValueError(
                    "Stem sample rates differ: expected %d Hz, got %d Hz for %s"
                    % (sample_rate, int(read_sample_rate), stem)
                )
            stems[stem] = audio

        if sample_rate is None or not stems:
            self._stems = {}
            return

        self._stems = stems
        self._sample_rate = int(sample_rate)
        self._frame_count = max((int(audio.shape[0]) for audio in stems.values()), default=0)
        with self._lock:
            self._gains = {stem: 1.0 for stem in stems}
            self._solo_stem = None
            self._cursor_frame = 0
        self._state = "stopped"
        self.state_changed.emit()

    def clear(self) -> None:
        self.stop()
        self._stems = {}
        self._sample_rate = 0
        self._frame_count = 0
        self._stream = None
        with self._lock:
            self._gains = {}
            self._solo_stem = None
            self._cursor_frame = 0
        self.state_changed.emit()

    def is_loaded(self) -> bool:
        return self._frame_count > 0 and bool(self._stems)

    def is_playing(self) -> bool:
        return self._state == "playing"

    def is_paused(self) -> bool:
        return self._state == "paused"

    def is_active(self) -> bool:
        return self._state in {"playing", "paused"}

    def play(self) -> None:
        if not self.is_loaded():
            return

        if self.position_ms() >= self.duration_ms():
            self.seek_ms(0)

        if sd is None:
            self.log.emit(
                "Playback failed: install the GUI audio dependency with `pip install sounddevice`."
            )
            return

        if self._stream is None:
            output_device = _sounddevice_output_device()
            if output_device is None:
                self.log.emit(
                    "Playback failed: PortAudio did not report any output devices. "
                    "Check your OS audio service or run `python -m sounddevice`."
                )
                return
            try:
                self._stream = sd.OutputStream(
                    device=output_device,
                    samplerate=self._sample_rate,
                    channels=2,
                    dtype="float32",
                    blocksize=4096,  # Changed from 1024 to fix WSL stuttering
                    latency="high",  # Changed from "low" to fix WSL stuttering
                    callback=self._audio_callback,
                )
            except Exception as exc:  # noqa: BLE001 - backend errors vary by platform.
                self.log.emit(f"Playback failed: could not open sounddevice stream: {exc}")
                return

        try:
            self._stream.start()
        except Exception as exc:  # noqa: BLE001 - backend errors vary by platform.
            self.log.emit(f"Playback failed: could not start sounddevice stream: {exc}")
            self._stream = None
            return

        self._state = "playing"
        self.state_changed.emit()

    def pause(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception as exc:  # noqa: BLE001 - backend errors vary by platform.
                self.log.emit(f"Playback pause failed: {exc}")
        self._state = "paused"
        self.state_changed.emit()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # noqa: BLE001 - backend errors vary by platform.
                self.log.emit(f"Playback stop failed: {exc}")
            self._stream = None
        with self._lock:
            self._cursor_frame = 0
        self._state = "stopped"
        self.state_changed.emit()

    def seek_ms(self, position_ms: int) -> None:
        if self._sample_rate <= 0:
            return
        frame = int((max(0, int(position_ms)) / 1000.0) * self._sample_rate)
        was_playing = self.is_playing()
        if was_playing and self._stream is not None:
            try:
                self._stream.stop()
            except Exception as exc:  # noqa: BLE001 - backend errors vary by platform.
                self.log.emit(f"Playback seek failed: {exc}")
                return
        with self._lock:
            self._cursor_frame = min(self._frame_count, max(0, frame))
        if was_playing and self._stream is not None:
            try:
                self._stream.start()
            except Exception as exc:  # noqa: BLE001 - backend errors vary by platform.
                self.log.emit(f"Playback seek failed: {exc}")

    def set_gain(self, stem: str, gain: float) -> None:
        if stem not in self._gains:
            return
        gain = min(1.0, max(0.0, float(gain)))
        with self._lock:
            if self._gains.get(stem) == gain:
                return
            self._gains[stem] = gain

    def set_solo_stem(self, stem: Optional[str]) -> None:
        with self._lock:
            if self._solo_stem == stem:
                return
            self._solo_stem = stem

    def position_ms(self) -> int:
        if self._sample_rate <= 0:
            return 0
        with self._lock:
            frame = int(self._cursor_frame)
        return int((frame / self._sample_rate) * 1000)

    def duration_ms(self) -> int:
        if self._sample_rate <= 0:
            return 0
        return int((self._frame_count / self._sample_rate) * 1000)

    def _audio_callback(self, outdata, frames: int, _time, status) -> None:
        if status:
            self.log.emit(f"Playback status: {status}")

        with self._lock:
            start_frame = int(self._cursor_frame)
            self._cursor_frame = min(self._frame_count, self._cursor_frame + int(frames))

        chunk = self._mixed_chunk(start_frame, int(frames))
        outdata[:] = chunk
        if start_frame + int(frames) >= self._frame_count:
            with self._lock:
                self._cursor_frame = self._frame_count
            self._state = "stopped"
            self.state_changed.emit()
            if sd is not None:
                raise sd.CallbackStop

    def _mixed_chunk(self, start_frame: int, frame_count: int) -> np.ndarray:
        mixed = np.zeros((frame_count, 2), dtype=np.float32)
        end_frame = start_frame + frame_count
        with self._lock:
            gains = dict(self._gains)
            solo_stem = self._solo_stem
        for stem, audio in self._stems.items():
            if solo_stem is not None and stem != solo_stem:
                continue

            gain = gains.get(stem, 1.0)
            if gain <= 0.0 or start_frame >= audio.shape[0]:
                continue

            stem_end = min(end_frame, int(audio.shape[0]))
            stem_frames = stem_end - start_frame
            if stem_frames > 0:
                mixed[:stem_frames] += audio[start_frame:stem_end] * gain

        return np.clip(mixed, -1.0, 1.0)


def _sounddevice_output_device() -> Optional[int]:
    if sd is None:
        return None

    try:
        devices = sd.query_devices()
    except Exception:  # noqa: BLE001 - backend errors vary by platform.
        return None

    try:
        default_device = sd.default.device
        default_output = int(default_device[1] if isinstance(default_device, (list, tuple)) else -1)
    except (TypeError, ValueError, IndexError):
        default_output = -1

    if default_output >= 0:
        try:
            default_info = devices[default_output]
            if int(default_info.get("max_output_channels", 0)) > 0:
                return default_output
        except (IndexError, TypeError, ValueError, AttributeError):
            pass

    for idx, info in enumerate(devices):
        try:
            if int(info.get("max_output_channels", 0)) > 0:
                return int(idx)
        except (TypeError, ValueError, AttributeError):
            continue

    return None


class DeviceSelector(QWidget):
    """Inline device picker without a popup window."""

    def __init__(self) -> None:
        super().__init__()
        self._items: list[tuple[QPushButton, str]] = []
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)

    def addItem(self, label: str, data: str) -> None:  # noqa: N802 - keep combo-like API.
        button = QPushButton(label)
        button.setObjectName("DeviceButton")
        button.setCheckable(True)
        button.setMinimumWidth(72)
        self._layout.addWidget(button)
        self._button_group.addButton(button)
        self._items.append((button, data))
        if len(self._items) == 1:
            button.setChecked(True)

    def currentData(self) -> str:  # noqa: N802 - keep combo-like API.
        for button, data in self._items:
            if button.isChecked():
                return data
        return self._items[0][1] if self._items else ""


class ThemedSpinBox(QSpinBox):
    """Spin box that redraws arrows after stylesheet customizations."""

    def __init__(self) -> None:
        super().__init__()
        self._arrow_color = QColor("#111827")

    def set_arrow_color(self, color_hex: str) -> None:
        self._arrow_color = QColor(color_hex)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override name.
        super().paintEvent(event)

        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        up_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_SpinBox,
            option,
            QStyle.SubControl.SC_SpinBoxUp,
            self,
        )
        down_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_SpinBox,
            option,
            QStyle.SubControl.SC_SpinBoxDown,
            self,
        )

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(self._arrow_color)
        painter.setPen(Qt.PenStyle.NoPen)
        self._draw_arrow(painter, up_rect, upward=True)
        self._draw_arrow(painter, down_rect, upward=False)

    def _draw_arrow(self, painter: QPainter, rect, upward: bool) -> None:
        center = rect.center()
        half_width = max(3, min(5, rect.width() // 3))
        half_height = max(2, min(4, rect.height() // 3))
        if upward:
            points = [
                QPoint(center.x(), center.y() - half_height),
                QPoint(center.x() - half_width, center.y() + half_height),
                QPoint(center.x() + half_width, center.y() + half_height),
            ]
        else:
            points = [
                QPoint(center.x() - half_width, center.y() - half_height),
                QPoint(center.x() + half_width, center.y() - half_height),
                QPoint(center.x(), center.y() + half_height),
            ]
        painter.drawPolygon(QPolygon(points))


class OscilloscopeWidget(QWidget):
    """Draw downsampled per-stem waveform traces."""

    seek_requested = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Oscilloscope")
        self.setMinimumHeight(170)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.traces: dict[str, np.ndarray] = {}
        self.levels: dict[str, float] = {}
        self.trace_colors = dict(TRACE_COLORS)
        self.theme_colors = dict(DEFAULT_THEME_COLORS)
        self.playhead_fraction = 0.0
        self._trace_paths: dict[str, QPainterPath] = {}
        self._cached_trace_rect = None
        self._trace_cache_valid = False

    def clear(self) -> None:
        self.traces.clear()
        self.levels.clear()
        self.playhead_fraction = 0.0
        self._invalidate_trace_cache()
        self.update()

    def set_traces(self, traces: dict[str, np.ndarray]) -> None:
        self.traces = traces
        self.levels = {stem: 1.0 for stem in traces}
        self._invalidate_trace_cache()
        self.update()

    def set_levels(self, levels: dict[str, float]) -> None:
        if self.levels == levels:
            return
        self.levels = levels
        self._invalidate_trace_cache()
        self.update()

    def set_playhead_fraction(self, fraction: float) -> None:
        fraction = min(1.0, max(0.0, float(fraction)))
        if self._playhead_pixel(fraction) == self._playhead_pixel(self.playhead_fraction):
            return
        self.playhead_fraction = fraction
        self.update()

    def set_colors(self, theme_colors: dict[str, str], trace_colors: dict[str, str]) -> None:
        self.theme_colors = dict(theme_colors)
        self.trace_colors = dict(trace_colors)
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override name.
        self._invalidate_trace_cache()
        super().resizeEvent(event)

    def _invalidate_trace_cache(self) -> None:
        self._trace_paths.clear()
        self._cached_trace_rect = None
        self._trace_cache_valid = False

    def _playhead_pixel(self, fraction: float) -> int:
        rect = self.rect().adjusted(10, 10, -10, -10)
        if rect.width() <= 0:
            return -1
        return rect.left() + int(min(1.0, max(0.0, float(fraction))) * rect.width())

    def _ensure_trace_paths(self, rect) -> None:
        if self._trace_cache_valid and self._cached_trace_rect == rect:
            return

        self._trace_paths.clear()
        self._cached_trace_rect = rect
        self._trace_cache_valid = True
        center_y = rect.center().y()
        half_height = max(1, rect.height() / 2 - 6)

        for stem in STEMS_4:
            trace = self.traces.get(stem)
            if trace is None or len(trace) < 2:
                continue

            level = self.levels.get(stem, 1.0)
            if level <= 0:
                continue

            points = len(trace)
            path = QPainterPath()
            path.moveTo(rect.left(), center_y - float(trace[0]) * level * half_height)
            for idx in range(1, points):
                x = rect.left() + int(idx * rect.width() / (points - 1))
                y = center_y - float(trace[idx]) * level * half_height
                path.lineTo(x, y)
            self._trace_paths[stem] = path

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override name.
        if not self.traces:
            return

        rect = self.rect().adjusted(10, 10, -10, -10)
        if rect.width() <= 0:
            return

        fraction = (event.position().x() - rect.left()) / rect.width()
        self.seek_requested.emit(min(1.0, max(0.0, float(fraction))))

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override name.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        base_color = QColor(self.theme_colors["field_bg"])
        painter.fillRect(self.rect(), base_color)

        rect = self.rect().adjusted(10, 10, -10, -10)
        guide_color = QColor(self.theme_colors["border"])
        painter.setPen(QPen(guide_color, 1))
        painter.drawLine(rect.left(), rect.center().y(), rect.right(), rect.center().y())

        if not self.traces:
            painter.setPen(QColor(self.theme_colors["muted_text"]))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Oscilloscope")
            return

        painter.setClipRect(rect)
        self._ensure_trace_paths(rect)

        for stem in STEMS_4:
            path = self._trace_paths.get(stem)
            if path is None:
                continue
            color = QColor(self.trace_colors.get(stem, self.theme_colors["text"]))
            trace_width = (
                OSCILLOSCOPE_TRACE_WIDTH_LIGHT
                if base_color.lightness() > 160
                else OSCILLOSCOPE_TRACE_WIDTH_DARK
            )
            painter.setPen(QPen(color, trace_width))
            painter.drawPath(path)

        playhead_x = rect.left() + int(self.playhead_fraction * rect.width())
        played_color = QColor(self.theme_colors["accent"])
        painter.setPen(QPen(played_color, OSCILLOSCOPE_PROGRESS_WIDTH))
        painter.drawLine(rect.left(), rect.bottom(), playhead_x, rect.bottom())
        playhead_color = QColor(self.theme_colors["text"])
        painter.setPen(QPen(playhead_color, OSCILLOSCOPE_PLAYHEAD_WIDTH))
        painter.drawLine(playhead_x, rect.top(), playhead_x, rect.bottom())


class MainWindow(QMainWindow):
    """Stemmy desktop front end."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Stemmy")
        self.setMinimumSize(860, 640)
        self.resize(860, 1500)
        ui_font_family = _load_font_family(Path(__file__).with_name(UI_FONT_FILE))
        self.setFont(QFont(ui_font_family or UI_FONT_FAMILY_FALLBACK))
        self.title_font_family = _load_font_family(Path(__file__).with_name("adrip1.ttf"))

        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[SeparationWorker] = None
        self.stem_paths: dict[str, Path] = {}
        self.volume_sliders: dict[str, QSlider] = {}
        self.solo_buttons: dict[str, QPushButton] = {}
        self.stem_labels: dict[str, QLabel] = {}
        self.color_buttons: dict[tuple[str, str], QPushButton] = {}
        self.theme_dialog: Optional[QDialog] = None
        self.settings = QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)
        self.theme_colors = _load_color_settings(
            self.settings,
            THEME_SETTINGS_GROUP,
            DEFAULT_THEME_COLORS,
        )
        self.trace_colors = _load_color_settings(
            self.settings,
            TRACE_SETTINGS_GROUP,
            TRACE_COLORS,
        )
        self.track_duration_ms = 0
        self.mixer = StemMixer(self)
        self.mixer.state_changed.connect(self._on_playback_state_changed)
        self.mixer.log.connect(self._append_log)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Choose an audio file")

        self.output_edit = QLineEdit(str((Path.cwd() / "separated").resolve()))

        self.device_combo = DeviceSelector()
        self.device_combo.addItem("CPU", "cpu")
        if torch.cuda.is_available():
            self.device_combo.addItem("CUDA", "cuda")
            for idx in range(torch.cuda.device_count()):
                self.device_combo.addItem(f"CUDA:{idx}", f"cuda:{idx}")

        self.chunk_spin = ThemedSpinBox()
        self.chunk_spin.setRange(0, 100_000)
        self.chunk_spin.setValue(256)
        self.chunk_spin.setSuffix(" frames")

        self.overlap_spin = ThemedSpinBox()
        self.overlap_spin.setRange(0, 100_000)
        self.overlap_spin.setValue(64)
        self.overlap_spin.setSuffix(" frames")

        self.amp_check = QCheckBox("AMP")
        self.amp_check.setEnabled(torch.cuda.is_available())

        self.run_button = QPushButton("Run Separation")
        self.run_button.clicked.connect(self.start_separation)

        self.theme_button = QPushButton("Theme")
        self.theme_button.setToolTip("Open theme editor")
        self.theme_button.clicked.connect(self._open_theme_editor)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

        self.outputs_list = QListWidget()
        self.outputs_list.currentItemChanged.connect(self._on_output_selection_changed)
        self.oscilloscope = OscilloscopeWidget()
        self.oscilloscope.seek_requested.connect(self._seek_stem_mix)
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

        self.playhead_timer = QTimer(self)
        self.playhead_timer.setInterval(33)
        self.playhead_timer.timeout.connect(self._refresh_playhead)

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
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        if self.title_font_family is not None:
            title.setFont(QFont(self.title_font_family, TITLE_FONT_SIZE, QFont.Weight.Bold))
        subtitle = QLabel("Separate an audio file into drums, bass, vocals, and other stems.")
        subtitle.setObjectName("Subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header_row.addLayout(title_block, 1)
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

        footer_row = QHBoxLayout()
        footer_row.addStretch(1)
        footer_row.addWidget(self.theme_button)
        layout.addLayout(footer_row)

    @Slot()
    def _apply_style(self) -> None:
        theme = self.theme_colors
        button_text = _contrast_text(theme["accent"])
        solo_text = _contrast_text(theme["solo"])
        disabled_bg = _blend_hex(theme["accent"], theme["field_bg"], 0.35)
        disabled_text = _blend_hex(theme["text"], theme["field_bg"], 0.55)
        slider_groove = _blend_hex(theme["text"], theme["field_bg"], 0.35)

        self._apply_window_palette()
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: %s;
                color: %s;
                font-size: 14px;
            }
            QLabel#Title {
                font-size: %dpx;
                font-weight: 700;
                color: %s;
            }
            QLabel#Subtitle {
                color: %s;
                font-size: 15px;
            }
            QGroupBox {
                background: %s;
                border: 1px solid %s;
                border-radius: 8px;
                margin-top: 10px;
                padding: 14px;
            }
            QFrame#MixerFrame {
                background: %s;
                border: 1px solid %s;
                border-radius: 8px;
            }
            QWidget#Oscilloscope {
                background: %s;
                border: 1px solid %s;
                border-radius: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
                color: %s;
                font-weight: 700;
            }
            QLabel#NowPlaying {
                color: %s;
            }
            QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QListWidget {
                background: %s;
                border: 1px solid %s;
                border-radius: 6px;
                color: %s;
                padding: 6px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background: %s;
                border: 0;
                width: 18px;
            }
            QSpinBox::up-button {
                border-top-right-radius: 5px;
            }
            QSpinBox::down-button {
                border-bottom-right-radius: 5px;
            }
            QLineEdit:disabled {
                background: %s;
                color: %s;
            }
            QPushButton {
                background: %s;
                color: %s;
                border: 0;
                border-radius: 6px;
                padding: 8px 14px;
                font-weight: 700;
            }
            QPushButton:disabled {
                background: %s;
                color: %s;
            }
            QPushButton:checked {
                background: %s;
                color: %s;
            }
            QPushButton#DeviceButton {
                background: %s;
                color: %s;
            }
            QPushButton#DeviceButton:checked {
                background: %s;
                color: %s;
            }
            QSlider::groove:vertical {
                background: %s;
                width: 2px;
                border-radius: 1px;
            }
            QSlider::handle:vertical {
                background: %s;
                border: 1px solid %s;
                width: 38px;
                height: 10px;
                margin: 0 -18px;
                border-radius: 4px;
            }
            QProgressBar {
                border: 1px solid %s;
                border-radius: 6px;
                height: 18px;
                text-align: center;
                background: %s;
                color: %s;
            }
            QProgressBar::chunk {
                background: %s;
                border-radius: 5px;
            }
            """
            % (
                theme["app_bg"],
                theme["text"],
                TITLE_FONT_SIZE,
                theme["accent"],
                theme["muted_text"],
                theme["panel_bg"],
                theme["border"],
                theme["app_bg"],
                theme["border"],
                theme["field_bg"],
                theme["border"],
                theme["accent"],
                theme["muted_text"],
                theme["field_bg"],
                theme["border"],
                theme["text"],
                theme["accent"],
                theme["panel_bg"],
                disabled_text,
                theme["accent"],
                button_text,
                disabled_bg,
                disabled_text,
                theme["solo"],
                solo_text,
                disabled_bg,
                disabled_text,
                theme["accent"],
                button_text,
                slider_groove,
                theme["accent"],
                theme["text"],
                theme["border"],
                theme["field_bg"],
                theme["text"],
                theme["accent"],
            )
        )
        self._refresh_color_widgets()
        self.oscilloscope.set_colors(self.theme_colors, self.trace_colors)
        self.chunk_spin.set_arrow_color(button_text)
        self.overlap_spin.set_arrow_color(button_text)

    def _apply_window_palette(self) -> None:
        palette = QPalette()
        theme = self.theme_colors
        palette.setColor(QPalette.ColorRole.Window, QColor(theme["app_bg"]))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(theme["text"]))
        palette.setColor(QPalette.ColorRole.Base, QColor(theme["field_bg"]))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(theme["panel_bg"]))
        palette.setColor(QPalette.ColorRole.Text, QColor(theme["text"]))
        palette.setColor(QPalette.ColorRole.Button, QColor(theme["accent"]))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(_contrast_text(theme["accent"])))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(theme["accent"]))
        palette.setColor(
            QPalette.ColorRole.HighlightedText,
            QColor(_contrast_text(theme["accent"])),
        )

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

    @Slot()
    def _open_theme_editor(self) -> None:
        if self.theme_dialog is None:
            self.color_buttons.clear()
            self.theme_dialog = QDialog(self)
            self.theme_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            self.theme_dialog.setWindowTitle("Theme Editor")
            self.theme_dialog.setMinimumWidth(720)
            self.theme_dialog.destroyed.connect(self._clear_theme_dialog)
            dialog_layout = QVBoxLayout(self.theme_dialog)
            dialog_layout.setContentsMargins(18, 18, 18, 18)
            dialog_layout.setSpacing(12)
            dialog_layout.addWidget(self._theme_widget())

            close_row = QHBoxLayout()
            close_row.addStretch(1)
            close_button = QPushButton("Close")
            close_button.clicked.connect(self._close_theme_editor)
            close_row.addWidget(close_button)
            dialog_layout.addLayout(close_row)

        self._refresh_color_widgets()
        self.theme_dialog.show()
        self.theme_dialog.raise_()
        self.theme_dialog.activateWindow()

    @Slot()
    def _close_theme_editor(self) -> None:
        if self.theme_dialog is not None:
            self.theme_dialog.close()

    @Slot()
    def _clear_theme_dialog(self) -> None:
        self.theme_dialog = None
        self.color_buttons.clear()

    def _theme_widget(self) -> QGroupBox:
        group = QGroupBox("Theme")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        theme_row = QHBoxLayout()
        for key, label in THEME_COLOR_LABELS.items():
            theme_row.addWidget(self._color_button("theme", key, label))
        theme_row.addStretch(1)
        layout.addLayout(theme_row)

        stem_row = QHBoxLayout()
        for stem in STEMS_4:
            stem_row.addWidget(self._color_button("trace", stem, stem.title()))
        stem_row.addStretch(1)
        layout.addLayout(stem_row)

        reset_row = QHBoxLayout()
        reset_row.addStretch(1)
        save_button = QPushButton("Save Theme")
        save_button.clicked.connect(self._save_theme_settings_with_status)
        reset_row.addWidget(save_button)

        reset_button = QPushButton("Restore Defaults")
        reset_button.clicked.connect(self._restore_default_theme)
        reset_row.addWidget(reset_button)
        layout.addLayout(reset_row)
        return group

    def _color_button(self, group: str, key: str, label: str) -> QPushButton:
        button = QPushButton(label)
        button.setFixedHeight(30)
        button.setMinimumWidth(72)
        button.setToolTip(f"Change {label} color")
        button.clicked.connect(
            lambda _checked=False, group=group, key=key: self._pick_color(group, key)
        )
        self.color_buttons[(group, key)] = button
        return button

    def _refresh_color_widgets(self) -> None:
        for (group, key), button in self.color_buttons.items():
            color = self.theme_colors[key] if group == "theme" else self.trace_colors[key]
            text_color = _contrast_text(color)
            button.setStyleSheet(
                "background: %s; color: %s; border: 1px solid %s;"
                "border-radius: 6px; padding: 6px 10px; font-weight: 700;"
                % (color, text_color, self.theme_colors["border"])
            )

        for stem, label in self.stem_labels.items():
            color = self.trace_colors.get(stem, self.theme_colors["text"])
            label.setStyleSheet("color: %s; font-weight: 700;" % color)

    def _pick_color(self, group: str, key: str) -> None:
        colors = self.theme_colors if group == "theme" else self.trace_colors
        current = QColor(colors[key])
        color = QColorDialog.getColor(current, self, "Choose color")
        if not color.isValid():
            return

        colors[key] = color.name(QColor.NameFormat.HexRgb)
        self._save_theme_settings()
        self._apply_style()

    @Slot()
    def _save_theme_settings(self) -> None:
        _save_color_settings(self.settings, THEME_SETTINGS_GROUP, self.theme_colors)
        _save_color_settings(self.settings, TRACE_SETTINGS_GROUP, self.trace_colors)
        self.settings.sync()

    @Slot()
    def _save_theme_settings_with_status(self) -> None:
        self._save_theme_settings()
        QMessageBox.information(self.theme_dialog or self, "Theme Saved", "Theme saved.")

    @Slot()
    def _restore_default_theme(self) -> None:
        self.theme_colors = dict(DEFAULT_THEME_COLORS)
        self.trace_colors = dict(TRACE_COLORS)
        self.settings.remove(THEME_SETTINGS_GROUP)
        self.settings.remove(TRACE_SETTINGS_GROUP)
        self.settings.sync()
        self._apply_style()

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
            self.stem_labels[stem] = stem_label

            slider = QSlider(Qt.Orientation.Vertical)
            slider.setRange(0, 100)
            slider.setValue(100)
            slider.setFixedHeight(140)
            slider.setFixedWidth(56)
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

    def _details_frame(self, title: str, widget: QWidget) -> QGroupBox:
        frame = QGroupBox(title)
        frame_layout = QVBoxLayout(frame)
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
        self.track_duration_ms = 0
        self.mixer.clear()
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
                item = QListWidgetItem(f"{stem}: {path}")
                item.setData(Qt.ItemDataRole.UserRole, path)
                self.outputs_list.addItem(item)
                loaded_count += 1
        if self.outputs_list.count() > 0:
            self.outputs_list.setCurrentRow(0)
        self.play_button.setEnabled(loaded_count > 0)
        self._load_oscilloscope_traces()
        self._load_mixer()
        self.play_button.setEnabled(self.mixer.is_loaded())
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
        if not self.mixer.is_loaded():
            QMessageBox.information(self, "No Stems Loaded", "Run separation before playing stems.")
            return

        if self.mixer.is_playing():
            self.mixer.pause()
            return

        if not self.mixer.is_paused():
            self.mixer.seek_ms(0)
        self.mixer.play()
        self.now_playing_label.setText("Playing stem mix")

    @Slot()
    def stop_playback(self) -> None:
        self.mixer.stop()

    @Slot(QListWidgetItem, QListWidgetItem)
    def _on_output_selection_changed(
        self,
        current: Optional[QListWidgetItem],
        _previous: Optional[QListWidgetItem],
    ) -> None:
        self.play_button.setEnabled(self.mixer.is_loaded())
        if current is not None and not self.mixer.is_playing():
            path = _item_path(current)
            if path is not None:
                self.now_playing_label.setText(path.name)

    @Slot()
    def _on_playback_state_changed(self) -> None:
        is_playing = self.mixer.is_playing()
        if is_playing and not self.playhead_timer.isActive():
            self.playhead_timer.start()
        elif not is_playing and self.playhead_timer.isActive():
            self.playhead_timer.stop()
            self._refresh_playhead()

        self.play_button.setText("Pause" if is_playing else "Play")
        self.stop_button.setEnabled(self.mixer.is_active())
        if not self.mixer.is_active() and self.stem_paths:
            self.now_playing_label.setText("Stem mix stopped")

    def _apply_mixer_volumes(self) -> None:
        soloed_stems = {stem for stem, button in self.solo_buttons.items() if button.isChecked()}
        levels: dict[str, float] = {}
        solo_stem = next(iter(soloed_stems), None)
        self.mixer.set_solo_stem(solo_stem)
        for stem in STEMS_4:
            slider = self.volume_sliders.get(stem)
            if slider is None:
                continue
            slider_volume = slider.value() / 100.0
            volume = slider_volume
            if soloed_stems and stem not in soloed_stems:
                volume = 0.0
            self.mixer.set_gain(stem, slider_volume)
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
        self.track_duration_ms = 0
        for stem in STEMS_4:
            path = self.stem_paths.get(stem)
            if path is None:
                continue
            try:
                traces[stem] = _load_waveform_trace(path)
                self.track_duration_ms = max(self.track_duration_ms, _audio_duration_ms(path))
            except (OSError, RuntimeError, ValueError) as exc:
                self._append_log(f"Could not load oscilloscope trace for {stem}: {exc}")
        self.oscilloscope.set_traces(traces)

    def _load_mixer(self) -> None:
        try:
            self.mixer.load_stems(self.stem_paths)
        except (OSError, RuntimeError, ValueError) as exc:
            self._append_log(f"Could not load stem mixer: {exc}")
            QMessageBox.warning(self, "Playback Setup Failed", str(exc))

    @Slot(float)
    def _seek_stem_mix(self, fraction: float) -> None:
        duration_ms = self._mix_duration_ms()
        if duration_ms <= 0:
            return

        position_ms = int(duration_ms * min(1.0, max(0.0, fraction)))
        self.mixer.seek_ms(position_ms)
        self.oscilloscope.set_playhead_fraction(position_ms / duration_ms)
        self.now_playing_label.setText(_format_position(position_ms, duration_ms))

    @Slot()
    def _refresh_playhead(self) -> None:
        duration_ms = self._mix_duration_ms()
        if duration_ms <= 0:
            self.oscilloscope.set_playhead_fraction(0.0)
            return

        position_ms = self._mix_position_ms()
        self.oscilloscope.set_playhead_fraction(position_ms / duration_ms)
        if self.mixer.is_playing() and position_ms >= duration_ms:
            self.stop_playback()

    def _mix_duration_ms(self) -> int:
        return max(self.track_duration_ms, self.mixer.duration_ms())

    def _mix_position_ms(self) -> int:
        return self.mixer.position_ms()


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


def _model_cache_key(
    checkpoint: Path,
    device: str,
    stem_count: int,
) -> tuple[Path, str, int, int, int]:
    stat = checkpoint.stat()
    return (
        checkpoint.expanduser().resolve(),
        str(device),
        int(stem_count),
        int(stat.st_mtime_ns),
        int(stat.st_size),
    )


def _load_waveform_trace(path: Path) -> np.ndarray:
    info = sf.info(str(path))
    if info.frames <= 0:
        raise ValueError("empty audio file")

    if info.frames <= OSCILLOSCOPE_POINTS:
        audio, _sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        mono = np.mean(audio, axis=1)
    else:
        indices = np.linspace(0, info.frames - 1, OSCILLOSCOPE_POINTS).astype(np.int64)
        mono = np.empty(len(indices), dtype=np.float32)
        with sf.SoundFile(str(path)) as audio_file:
            for idx, frame in enumerate(indices):
                audio_file.seek(int(frame))
                sample = audio_file.read(frames=1, dtype="float32", always_2d=True)
                mono[idx] = float(np.mean(sample)) if sample.size else 0.0

    peak = float(np.max(np.abs(mono)))
    if peak > 0:
        mono = mono / peak
    return mono.astype(np.float32, copy=False)


def _read_stem_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.size == 0 or audio.shape[0] <= 0:
        raise ValueError(f"empty audio file: {path}")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)


def _audio_duration_ms(path: Path) -> int:
    info = sf.info(str(path))
    if info.samplerate <= 0:
        return 0
    return int((info.frames / info.samplerate) * 1000)


def _format_position(position_ms: int, duration_ms: int) -> str:
    return "%s / %s" % (_format_time(position_ms), _format_time(duration_ms))


def _format_time(milliseconds: int) -> str:
    total_seconds = max(0, int(milliseconds / 1000))
    minutes, seconds = divmod(total_seconds, 60)
    return "%d:%02d" % (minutes, seconds)


def _item_path(item: QListWidgetItem) -> Optional[Path]:
    raw_path = item.data(Qt.ItemDataRole.UserRole)
    if raw_path is None:
        return None
    return Path(str(raw_path)).expanduser().resolve()


def _load_color_settings(
    settings: QSettings,
    group: str,
    defaults: dict[str, str],
) -> dict[str, str]:
    colors = dict(defaults)
    settings.beginGroup(group)
    try:
        for key, default in defaults.items():
            raw_value = settings.value(key, default)
            color = QColor(str(raw_value))
            if color.isValid():
                colors[key] = color.name(QColor.NameFormat.HexRgb)
    finally:
        settings.endGroup()
    return colors


def _save_color_settings(settings: QSettings, group: str, colors: dict[str, str]) -> None:
    settings.beginGroup(group)
    try:
        for key, color_hex in colors.items():
            color = QColor(color_hex)
            if color.isValid():
                settings.setValue(key, color.name(QColor.NameFormat.HexRgb))
    finally:
        settings.endGroup()


def _contrast_text(background_hex: str) -> str:
    color = QColor(background_hex)
    luminance = 0.299 * color.redF() + 0.587 * color.greenF() + 0.114 * color.blueF()
    return "#111827" if luminance > 0.62 else "#f9fafb"


def _blend_hex(foreground_hex: str, background_hex: str, amount: float) -> str:
    amount = min(1.0, max(0.0, amount))
    foreground = QColor(foreground_hex)
    background = QColor(background_hex)
    red = round(background.red() + (foreground.red() - background.red()) * amount)
    green = round(background.green() + (foreground.green() - background.green()) * amount)
    blue = round(background.blue() + (foreground.blue() - background.blue()) * amount)
    return QColor(red, green, blue).name(QColor.NameFormat.HexRgb)


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

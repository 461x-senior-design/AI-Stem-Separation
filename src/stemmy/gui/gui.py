#!/usr/bin/env python3
"""PySide6 front end for Stemmy source separation."""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import torch
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from stemmy.constants import STEMS_4
from stemmy.inference import (
    InferenceConfig,
    config_from_checkpoint,
    load_pth_model,
    load_torchscript_model,
    separate_audio_file,
)
from stemmy.logging_config import setup_logging


@dataclass(frozen=True)
class SeparationJob:
    """User-selected separation options."""

    input_file: Path
    output_dir: Path
    checkpoint: Optional[Path]
    torchscript: Optional[Path]
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

        if self.job.torchscript is not None:
            model = load_torchscript_model(self.job.torchscript, device=self.job.device)
            return model, InferenceConfig(), None

        raise ValueError("Choose either a .pth checkpoint or a .pt TorchScript model.")


class MainWindow(QMainWindow):
    """Stemmy desktop front end."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Stemmy")
        self.setMinimumSize(860, 640)

        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[SeparationWorker] = None

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Choose an audio file")

        self.output_edit = QLineEdit(str((Path.cwd() / "separated").resolve()))
        self.checkpoint_edit = QLineEdit(str(_default_model_path(".pth") or ""))
        self.torchscript_edit = QLineEdit(str(_default_model_path(".pt") or ""))

        self.checkpoint_radio = QRadioButton("Checkpoint (.pth)")
        self.torchscript_radio = QRadioButton("TorchScript (.pt)")
        self.checkpoint_radio.setChecked(True)

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
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Run details appear here.")

        self._build_ui()
        self._connect_mode_controls()
        self._apply_style()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        header_row = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("Stemmy")
        title.setObjectName("Title")
        subtitle = QLabel("Separate an audio file into drums, bass, vocals, and other stems.")
        subtitle.setObjectName("Subtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header_row.addLayout(title_block, 1)
        header_row.addWidget(self.dark_mode_check, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header_row)

        files_group = QGroupBox("Files")
        files_form = QFormLayout(files_group)
        files_form.addRow("Input audio", self._path_picker_row(self.input_edit, self._pick_input))
        files_form.addRow(
            "Output folder",
            self._path_picker_row(self.output_edit, self._pick_output_dir),
        )
        layout.addWidget(files_group)

        model_group = QGroupBox("Model")
        model_layout = QGridLayout(model_group)
        model_layout.addWidget(self.checkpoint_radio, 0, 0)
        model_layout.addWidget(
            self._path_picker_row(self.checkpoint_edit, self._pick_checkpoint),
            0,
            1,
        )
        model_layout.addWidget(self.torchscript_radio, 1, 0)
        model_layout.addWidget(
            self._path_picker_row(self.torchscript_edit, self._pick_torchscript),
            1,
            1,
        )
        model_layout.setColumnStretch(1, 1)
        layout.addWidget(model_group)

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

        details = QHBoxLayout()
        outputs_frame = self._details_frame("Output Stems", self.outputs_list)
        log_frame = self._details_frame("Log", self.log_view)
        details.addWidget(outputs_frame, 1)
        details.addWidget(log_frame, 2)
        layout.addLayout(details, 1)

    def _connect_mode_controls(self) -> None:
        self.checkpoint_radio.toggled.connect(self._update_model_mode)
        self.torchscript_radio.toggled.connect(self._update_model_mode)
        self._update_model_mode()

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
                font-size: 30px;
                font-weight: 700;
                color: #f9fafb;
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
                background: #7FE80E;
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
            QProgressBar {
                border: 1px solid #4b5563;
                border-radius: 6px;
                height: 18px;
                text-align: center;
                background: #0f172a;
                color: #f9fafb;
            }
            QProgressBar::chunk {
                background: #7FE80E;
                border-radius: 5px;
            }
            """
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
                font-size: 30px;
                font-weight: 700;
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
            QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QListWidget {
                background: #ffffff;
                border: 1px solid #bcccdc;
                border-radius: 6px;
                padding: 6px;
            }
            QPushButton {
                background: #7FE80E;
                color: #1f2933;
                border: 0;
                border-radius: 6px;
                padding: 8px 14px;
                font-weight: 700;
            }
            QPushButton:disabled {
                background: #b7d69a;
                color: #52606d;
            }
            QProgressBar {
                border: 1px solid #bcccdc;
                border-radius: 6px;
                height: 18px;
                text-align: center;
                background: #ffffff;
            }
            QProgressBar::chunk {
                background: #7FE80E;
                border-radius: 5px;
            }
            """
        )

    def _apply_window_palette(self, dark: bool) -> None:
        palette = QPalette()
        if dark:
            palette.setColor(QPalette.ColorRole.Window, QColor("#111827"))
            palette.setColor(QPalette.ColorRole.WindowText, QColor("#e5e7eb"))
            palette.setColor(QPalette.ColorRole.Base, QColor("#0f172a"))
            palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#1f2937"))
            palette.setColor(QPalette.ColorRole.Text, QColor("#f9fafb"))
            palette.setColor(QPalette.ColorRole.Button, QColor("#7FE80E"))
            palette.setColor(QPalette.ColorRole.ButtonText, QColor("#111827"))
            palette.setColor(QPalette.ColorRole.Highlight, QColor("#7FE80E"))
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#111827"))
        else:
            palette.setColor(QPalette.ColorRole.Window, QColor("#f5f6f8"))
            palette.setColor(QPalette.ColorRole.WindowText, QColor("#1f2933"))
            palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
            palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f5f6f8"))
            palette.setColor(QPalette.ColorRole.Text, QColor("#1f2933"))
            palette.setColor(QPalette.ColorRole.Button, QColor("#7FE80E"))
            palette.setColor(QPalette.ColorRole.ButtonText, QColor("#1f2933"))
            palette.setColor(QPalette.ColorRole.Highlight, QColor("#7FE80E"))
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#1f2933"))

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
    def _update_model_mode(self) -> None:
        use_checkpoint = self.checkpoint_radio.isChecked()
        self.checkpoint_edit.setEnabled(use_checkpoint)
        self.torchscript_edit.setEnabled(not use_checkpoint)

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
    def _pick_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose checkpoint",
            str(Path.cwd()),
            "PyTorch Checkpoints (*.pth);;All Files (*)",
        )
        if path:
            self.checkpoint_edit.setText(path)
            self.checkpoint_radio.setChecked(True)

    @Slot()
    def _pick_torchscript(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose TorchScript model",
            str(Path.cwd()),
            "TorchScript Models (*.pt);;All Files (*)",
        )
        if path:
            self.torchscript_edit.setText(path)
            self.torchscript_radio.setChecked(True)

    @Slot()
    def start_separation(self) -> None:
        job = self._build_job()
        if job is None:
            return

        self.outputs_list.clear()
        self.log_view.clear()
        self._set_running(True)
        self._append_log(f"Input: {job.input_file}")
        self._append_log(f"Output: {job.output_dir}")
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

            checkpoint = None
            torchscript = None
            if self.checkpoint_radio.isChecked():
                checkpoint = _validated_file(self.checkpoint_edit.text(), "Checkpoint")
            else:
                torchscript = _validated_file(self.torchscript_edit.text(), "TorchScript model")

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
                torchscript=torchscript,
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
        for stem in list(STEMS_4):
            path = paths.get(stem)
            if path:
                self.outputs_list.addItem(f"{stem}: {path}")
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


def _default_model_path(suffix: str) -> Optional[Path]:
    candidates: list[Path] = []
    candidates.extend(Path("runs/best_ckpt").glob(f"*{suffix}"))
    candidates.extend(Path("best").glob(f"*{suffix}"))
    candidates.extend(Path("checkpoints").glob(f"*{suffix}"))

    files = [path.resolve() for path in candidates if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


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

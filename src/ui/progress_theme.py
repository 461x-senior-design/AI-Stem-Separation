import threading
from datetime import timedelta
from itertools import cycle

from rich.progress import BarColumn, Progress, ProgressColumn, TextColumn
from rich.table import Column
from rich.text import Text

from src.stemmy.constants import (
    BAR_COMPLETE_STYLE,
    BAR_FINISHED_STYLE,
    BAR_PULSE_STYLE,
    COUNT_STYLE,
    ELAPSED_STYLE,
    EQ_FRAMES,
    EQ_STYLE,
    LOSS_STYLE,
    REMAINING_STYLE,
)


class ThemedTimeElapsedColumn(ProgressColumn):
    """Elapsed time with explicit themed color."""

    def __init__(self, style=ELAPSED_STYLE, table_column=None):
        super().__init__(table_column=table_column)
        self.style = style

    def render(self, task):
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("-:--:--", style=self.style)
        return Text(str(timedelta(seconds=max(0, int(elapsed)))), style=self.style)


class ThemedTimeRemainingColumn(ProgressColumn):
    """Remaining ETA with explicit themed color."""

    def __init__(self, style=REMAINING_STYLE, table_column=None):
        super().__init__(table_column=table_column)
        self.style = style

    def render(self, task):
        if task.total is None:
            return Text("", style=self.style)
        remaining = task.time_remaining
        if remaining is None:
            return Text("-:--:--", style=self.style)
        minutes, seconds = divmod(int(remaining), 60)
        hours, minutes = divmod(minutes, 60)
        return Text(f"{hours:d}:{minutes:02d}:{seconds:02d}", style=self.style)


class ThemedStepColumn(ProgressColumn):
    """Render setup step text with explicit style (no markup)."""

    def __init__(self, table_column=None):
        super().__init__(table_column=table_column)

    def render(self, task):
        value = task.fields.get("step", "")
        value_style = task.fields.get("step_style", LOSS_STYLE)
        return Text(str(value), style=value_style)


def start_eq_animator(progress, task_id, fps=24):
    """Animate EQ text at a fixed rate, independent of task advance updates."""
    eq_frames = cycle(EQ_FRAMES)
    stop_event = threading.Event()

    def _run():
        while not stop_event.wait(1.0 / fps):
            # Keep EQ status fixed-width while preserving the full frame shape.
            progress.update(task_id, eq=f"{next(eq_frames):<6}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return stop_event, thread


def create_themed_progress(title_template, title_style, refresh_per_second=30):
    """Create a themed progress instance shared across modules."""
    return Progress(
        TextColumn(title_template, style=title_style, table_column=Column(width=15)),
        BarColumn(
            complete_style=BAR_COMPLETE_STYLE,
            finished_style=BAR_FINISHED_STYLE,
            pulse_style=BAR_PULSE_STYLE,
            bar_width=34,
            table_column=Column(width=34),
        ),
        TextColumn(
            "{task.completed}/{task.total}", style=COUNT_STYLE, table_column=Column(width=8)
        ),
        ThemedTimeElapsedColumn(style=ELAPSED_STYLE, table_column=Column(width=9)),
        ThemedTimeRemainingColumn(style=REMAINING_STYLE, table_column=Column(width=9)),
        TextColumn(
            "{task.fields[eq]}",
            style=EQ_STYLE,
            table_column=Column(width=10, justify="left"),
        ),
        refresh_per_second=refresh_per_second,
    )


def create_setup_progress(title_template, title_style, refresh_per_second=30):
    """Create a themed setup progress instance for pre-training initialization."""
    return Progress(
        TextColumn(title_template, style=title_style, table_column=Column(width=15)),
        BarColumn(
            complete_style=BAR_COMPLETE_STYLE,
            finished_style=BAR_FINISHED_STYLE,
            pulse_style=BAR_PULSE_STYLE,
            bar_width=34,
            table_column=Column(width=34),
        ),
        TextColumn(
            "{task.completed}/{task.total}", style=COUNT_STYLE, table_column=Column(width=8)
        ),
        ThemedTimeElapsedColumn(style=ELAPSED_STYLE, table_column=Column(width=9)),
        ThemedTimeRemainingColumn(style=REMAINING_STYLE, table_column=Column(width=9)),
        ThemedStepColumn(table_column=Column(width=28)),
        refresh_per_second=refresh_per_second,
    )

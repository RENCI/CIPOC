"""Live terminal progress displays for LangGraph agent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import shutil
import sys
import time
from typing import Any, Literal, Mapping, TextIO

from langgraph.graph.state import CompiledStateGraph


TaskKind = Literal["llm", "deterministic", "container"]

_DEFAULT_NODE_KINDS: dict[str, TaskKind] = {
    "initialize": "deterministic",
    "retrieve_clinical_notes": "deterministic",
    "extract_group_values": "llm",
    "variable_branch": "container",
    "extract_individual_value": "llm",
    "validate_extraction": "deterministic",
    "repair_invalid_extraction": "llm",
    "complete_variable": "container",
    "merge_variable_results": "deterministic",
    "summarize_note": "llm",
    "check_note_for_cancer": "llm",
    "get_cancer_mentions": "llm",
}


class _Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    MAIN = "\033[38;5;74m"
    WHITE = "\033[97m"
    GREY = "\033[90m"
    SUCCESS = "\033[38;5;33m"
    ERROR = "\033[38;5;208m"
    ACTIVE = "\033[36m"
    LLM = "\033[38;5;186m"


@dataclass
class _Task:
    task_id: str
    node_name: str
    label: str
    kind: TaskKind
    namespace: tuple[str, ...]
    started_at: float
    status: str = "active"
    finished_at: float | None = None
    error: str | None = None
    branch_key: str | None = None


@dataclass
class _Branch:
    key: str
    item_id: str
    name: str
    status: str = "pending"
    task_ids: list[str] = field(default_factory=list)
    value: str | None = None
    errors: list[str] = field(default_factory=list)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _task_label(namespace: tuple[str, ...], node_name: str) -> str:
    scope = [part.split(":", maxsplit=1)[0] for part in namespace]
    names = [*scope, node_name]
    return " > ".join(_humanize(name) for name in names)


def _humanize(name: str) -> str:
    return name.replace("_", " ").title()


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def _truncate(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width == 1:
        return "…"
    return value[: width - 1] + "…"


def _progress_bar(completed: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "░" * width
    filled = width if completed >= total else int(width * completed / total)
    return "█" * filled + "░" * (width - filled)


def _expected_step_count(
    graph: CompiledStateGraph,
    node_kinds: Mapping[str, TaskKind] | None,
) -> int | None:
    kinds = {**_DEFAULT_NODE_KINDS, **(node_kinds or {})}
    try:
        node_names = graph.get_graph().nodes
    except Exception:
        return None
    return sum(
        name not in {"__start__", "__end__"}
        and kinds.get(name, "deterministic") != "container"
        for name in node_names
    )


class _ProgressDisplay:
    """Render a compact live dashboard without requiring a UI dependency."""

    def __init__(
        self,
        description: str,
        *,
        node_kinds: Mapping[str, TaskKind] | None = None,
        expected_steps: int | None = None,
        stream: TextIO | None = None,
    ):
        self.description = description
        self.node_kinds = {**_DEFAULT_NODE_KINDS, **(node_kinds or {})}
        self.expected_steps = expected_steps
        self.stream = stream or sys.stdout
        self.started_at = time.monotonic()
        self.tasks: dict[str, _Task] = {}
        self._order: list[str] = []
        self._draw_count = 0
        self._rendered_lines = 0
        self._notebook = _in_notebook()
        self._interactive = self._notebook or bool(
            getattr(self.stream, "isatty", lambda: False)()
        )
        self._color = self._interactive and "NO_COLOR" not in os.environ
        self._fatal_error: str | None = None

    def _style(self, text: str, *styles: str) -> str:
        if not self._color:
            return text
        return "".join(styles) + text + _Color.RESET

    def _kind(self, node_name: str) -> TaskKind:
        return self.node_kinds.get(node_name, "deterministic")

    def start(
        self,
        task_id: str,
        namespace: tuple[str, ...],
        node_name: str,
        task_input: Any,
    ) -> None:
        task = _Task(
            task_id=task_id,
            node_name=node_name,
            label=_task_label(namespace, node_name),
            kind=self._kind(node_name),
            namespace=namespace,
            started_at=time.monotonic(),
        )
        self.tasks[task_id] = task
        self._order.append(task_id)
        self._task_started(task, task_input)
        if self._interactive:
            self.draw()
        elif task.kind != "container":
            self._write(f"  > {task.label} [{task.kind}]")

    def finish(
        self,
        task_id: str,
        namespace: tuple[str, ...],
        node_name: str,
        result: Any,
        error: Any = None,
    ) -> None:
        task = self.tasks.get(task_id)
        if task is None:
            task = _Task(
                task_id=task_id,
                node_name=node_name,
                label=_task_label(namespace, node_name),
                kind=self._kind(node_name),
                namespace=namespace,
                started_at=time.monotonic(),
            )
            self.tasks[task_id] = task
            self._order.append(task_id)

        task.finished_at = time.monotonic()
        task.status = "error" if error is not None else "success"
        task.error = str(error) if error is not None else None
        self._task_finished(task, result)
        if self._interactive:
            self.draw()
        elif task.kind != "container":
            marker = "x" if task.status in {"error", "invalid"} else "v"
            detail = f": {task.error}" if task.error else ""
            self._write(f"  {marker} {task.label}{detail}")

    def _task_started(self, task: _Task, task_input: Any) -> None:
        pass

    def _task_finished(self, task: _Task, result: Any) -> None:
        pass

    def fail(self, error: BaseException) -> None:
        self._fatal_error = str(error)
        now = time.monotonic()
        for task in self.tasks.values():
            if task.status == "active":
                task.status = "error"
                task.finished_at = now
                task.error = "Run stopped before this task completed."
        self.draw(final=True)

    def complete(self) -> None:
        self.draw(final=True)

    def _width(self) -> int:
        return max(60, min(shutil.get_terminal_size((100, 24)).columns, 120))

    def _duration(self, task: _Task) -> str:
        end = task.finished_at or time.monotonic()
        elapsed = end - task.started_at
        return f"{elapsed:.1f}s" if elapsed >= 0.1 else "<0.1s"

    def _kind_symbol(self, task: _Task) -> str:
        if task.kind == "llm":
            return self._style("◉", _Color.LLM)
        return self._style("◆", _Color.MAIN)

    def _status_symbol(self, task: _Task) -> str:
        if task.status == "success":
            return self._style("✓", _Color.SUCCESS)
        if task.status in {"error", "invalid"}:
            return self._style("✗", _Color.ERROR)
        return self._style("●", _Color.ACTIVE, _Color.BOLD)

    def _lane(self) -> str:
        visible_ids = [
            task_id
            for task_id in self._order
            if self.tasks[task_id].kind != "container"
        ][-5:]
        parts: list[str] = []
        for task_id in visible_ids:
            task = self.tasks[task_id]
            leaf_label = task.label.rsplit(" > ", maxsplit=1)[-1]
            parts.append(f"{self._status_symbol(task)} {_truncate(leaf_label, 20)}")

        separator = self._style("  →  ", _Color.GREY)
        lane = separator.join(parts)
        if len(self._order) > len(visible_ids):
            lane = self._style("…  ", _Color.GREY) + lane
        return "  " + lane

    def _render(self, final: bool) -> list[str]:
        width = self._width()
        elapsed = time.monotonic() - self.started_at
        tasks = [task for task in self.tasks.values() if task.kind != "container"]
        succeeded = sum(task.status == "success" for task in tasks)
        failed = sum(task.status in {"error", "invalid"} for task in tasks)
        active = sum(task.status == "active" for task in tasks)

        title = (
            f"  {self._style('CIPOC', _Color.BOLD, _Color.MAIN)}"
            f"{self._style(' / ' + self.description.upper(), _Color.GREY)}"
        )
        stats = [f"{succeeded} complete"]
        if failed:
            stats.append(self._style(f"{failed} failed", _Color.ERROR))
        if active and not final:
            stats.append(self._style(f"{active} running", _Color.ACTIVE))
        stats.append(f"{elapsed:.1f}s")

        completed = succeeded + failed
        total = completed if final else self.expected_steps or completed + active
        activity = _progress_bar(completed, total)
        progress = (
            f"  [{self._style(activity, _Color.MAIN)}]"
            f"  {completed}/{total}"
            f"  {self._style('  '.join(stats), _Color.WHITE)}"
        )

        lines = ["", title, "", progress, "", self._lane(), ""]
        lines.append("  " + self._style("━" * (width - 4), _Color.GREY))

        recent_complete = [
            task_id
            for task_id in self._order
            if self.tasks[task_id].kind != "container"
            and self.tasks[task_id].status != "active"
        ][-5:]
        active_ids = [
            task_id
            for task_id in self._order
            if self.tasks[task_id].kind != "container"
            and self.tasks[task_id].status == "active"
        ]
        visible_ids = recent_complete + active_ids

        if not visible_ids:
            lines.append(
                f"  {self._style('●', _Color.ACTIVE)} "
                f"{self._style('Starting…', _Color.DIM)}"
            )

        label_width = max(20, width - 24)
        for task_id in visible_ids:
            task = self.tasks[task_id]
            label = _truncate(task.label, label_width)
            timing = (
                self._style("running", _Color.ACTIVE)
                if task.status == "active"
                else self._style(self._duration(task), _Color.DIM)
            )
            padded_label = f"{label:<{label_width}}"
            lines.append(
                f"  {self._status_symbol(task)} {self._style(padded_label, _Color.WHITE)}"
                f"  {self._kind_symbol(task)} {timing}"
            )
            if task.error:
                lines.append(
                    f"    {self._style('↳ ' + _truncate(task.error, width - 8), _Color.ERROR)}"
                )

        lines.extend(self._completion_lines(final, elapsed, succeeded + failed))
        return lines

    def _completion_lines(self, final: bool, elapsed: float, count: int) -> list[str]:
        if self._fatal_error:
            return [
                "",
                f"  {self._style('✗ Run failed', _Color.BOLD, _Color.ERROR)}"
                f"  {self._style(_truncate(self._fatal_error, self._width() - 20), _Color.ERROR)}",
            ]
        if final:
            step_word = "step" if count == 1 else "steps"
            return [
                "",
                f"  {self._style('Done.', _Color.BOLD, _Color.SUCCESS)}"
                f"  {count} {step_word} in {elapsed:.1f}s",
            ]
        return []

    def draw(self, *, final: bool = False) -> None:
        if not self._interactive:
            if self._draw_count == 0:
                self._write(f"CIPOC / {self.description}")
            if final:
                elapsed = time.monotonic() - self.started_at
                failed = sum(
                    task.status in {"error", "invalid"}
                    for task in self.tasks.values()
                    if task.kind != "container"
                )
                finished = sum(
                    task.status != "active"
                    for task in self.tasks.values()
                    if task.kind != "container"
                )
                if self._fatal_error:
                    self._write(f"  failed after {elapsed:.1f}s: {self._fatal_error}")
                else:
                    suffix = f", {failed} failed" if failed else ""
                    step_word = "step" if finished == 1 else "steps"
                    self._write(
                        f"  complete: {finished} {step_word}{suffix} in {elapsed:.1f}s"
                    )
            self._draw_count += 1
            return

        lines = self._render(final)
        if self._notebook:
            from IPython.display import clear_output

            clear_output(wait=True)
            self._write("\n".join(lines))
        else:
            self._redraw_terminal(lines)
        self._draw_count += 1

    def _redraw_terminal(self, lines: list[str]) -> None:
        line_count = max(len(lines), self._rendered_lines)
        if self._rendered_lines:
            self.stream.write(f"\033[{self._rendered_lines}A")
        for index in range(line_count):
            line = lines[index] if index < len(lines) else ""
            self.stream.write(f"\r\033[2K{line}\n")
        self.stream.flush()
        self._rendered_lines = line_count

    def _write(self, text: str) -> None:
        self.stream.write(text + "\n")
        self.stream.flush()


class _BranchProgressDisplay(_ProgressDisplay):
    """Render shared graph work and dynamic fan-out branches separately."""

    def __init__(
        self,
        description: str,
        graph_input: Any,
        *,
        node_kinds: Mapping[str, TaskKind] | None = None,
        stream: TextIO | None = None,
    ):
        super().__init__(description, node_kinds=node_kinds, stream=stream)
        self.branches: dict[str, _Branch] = {}
        self._branch_order: list[str] = []
        self._runtime_branches: dict[str, str] = {}
        self._activities: list[tuple[float, str, str]] = []
        self._load_requested_variables(graph_input)

    def _load_requested_variables(self, graph_input: Any) -> None:
        requested = _field(graph_input, "requested_variables")
        variables = _field(requested, "variables", []) or []
        for index, variable in enumerate(variables):
            item_id = str(_field(variable, "item_id", index + 1))
            name = str(_field(variable, "name") or f"Variable {item_id}")
            key = item_id
            self.branches[key] = _Branch(key=key, item_id=item_id, name=name)
            self._branch_order.append(key)

    def _variable_from_input(self, task_input: Any) -> Any:
        task = _field(task_input, "task")
        return _field(task, "variable") if task is not None else _field(task_input, "variable")

    def _branch_for_variable(self, variable: Any) -> _Branch | None:
        if variable is None:
            return None
        item_id = str(_field(variable, "item_id", ""))
        if not item_id:
            return None
        if item_id not in self.branches:
            name = str(_field(variable, "name") or f"Variable {item_id}")
            self.branches[item_id] = _Branch(
                key=item_id,
                item_id=item_id,
                name=name,
            )
            self._branch_order.append(item_id)
        return self.branches[item_id]

    def _runtime_id(self, task: _Task) -> str | None:
        if task.namespace:
            root = task.namespace[0]
            if ":" in root:
                return root.split(":", maxsplit=1)[1]
        if task.node_name == "variable_branch":
            return task.task_id
        return None

    def _task_started(self, task: _Task, task_input: Any) -> None:
        runtime_id = self._runtime_id(task)
        branch = None
        if runtime_id is not None:
            branch_key = self._runtime_branches.get(runtime_id)
            branch = self.branches.get(branch_key) if branch_key else None
        if branch is None:
            branch = self._branch_for_variable(self._variable_from_input(task_input))
        if branch is None:
            return

        task.branch_key = branch.key
        branch.status = "active"
        if runtime_id is not None:
            self._runtime_branches[runtime_id] = branch.key
        if task.kind != "container":
            branch.task_ids.append(task.task_id)

        if task.node_name == "repair_invalid_extraction":
            details = self._validation_errors(task_input)
            detail = f": {details[0]}" if details else ""
            self._activity(f"{branch.name} repair started{detail}", "retry")

    def _task_finished(self, task: _Task, result: Any) -> None:
        if task.branch_key is None:
            return
        branch = self.branches[task.branch_key]

        if task.node_name == "validate_extraction":
            errors = self._validation_errors(result)
            if errors:
                task.status = "invalid"
                task.error = errors[0]
                branch.errors = errors
                self._activity(
                    f"{branch.name} validation rejected: {errors[0]}",
                    "error",
                )

        output = self._variable_result(result)
        if output is not None:
            value = _field(output, "value")
            branch.value = "No value" if value is None else str(value)
            errors = list(_field(output, "validation_errors", []) or [])
            is_valid = _field(output, "is_valid", True)
            branch.errors = errors
            branch.status = "success" if is_valid else "error"

        if task.status == "error":
            branch.status = "error"
            if task.error:
                branch.errors = [task.error]

        if task.node_name == "variable_branch" and task.status == "success":
            if branch.status == "active":
                branch.status = "success"
            if branch.status == "success":
                result_text = f" as {branch.value}" if branch.value is not None else ""
                self._activity(f"{branch.name} resolved{result_text}", "success")

    def _validation_errors(self, value: Any) -> list[str]:
        task = _field(value, "task", value)
        return [str(error) for error in (_field(task, "validation_errors", []) or [])]

    def _variable_result(self, result: Any) -> Any:
        outputs = _field(result, "variable_results")
        if outputs:
            return outputs[0]
        outputs = _field(result, "variables")
        if outputs:
            return outputs[0]
        return None

    def _activity(self, text: str, status: str) -> None:
        self._activities.append((time.monotonic() - self.started_at, text, status))

    def fail(self, error: BaseException) -> None:
        for branch in self.branches.values():
            if branch.status == "active":
                branch.status = "error"
                branch.errors = ["Run stopped before this branch completed."]
        super().fail(error)

    def _operation_label(self, node_name: str) -> str:
        labels = {
            "extract_individual_value": "Extract",
            "validate_extraction": "Validate",
            "repair_invalid_extraction": "Repair",
        }
        return labels.get(node_name, _humanize(node_name))

    def _operation_line(self, task: _Task, prefix: str, width: int) -> list[str]:
        if task.node_name == "repair_invalid_extraction":
            status = self._style("↻", _Color.LLM)
        else:
            status = self._status_symbol(task)
        timing = (
            self._style(self._duration(task), _Color.DIM)
            if task.status != "active"
            else self._style(f"running {self._duration(task)}", _Color.ACTIVE)
        )
        label = self._operation_label(task.node_name)
        line = (
            f"  {prefix} {status} {self._style(label, _Color.WHITE)} "
            f"{self._kind_symbol(task)}  {timing}"
        )
        lines = [line]
        if task.error:
            lines.append(
                f"  {prefix}   {self._style('↳ ' + _truncate(task.error, width - 10), _Color.ERROR)}"
            )
        return lines

    def _render(self, final: bool) -> list[str]:
        width = self._width()
        elapsed = time.monotonic() - self.started_at
        branches = [self.branches[key] for key in self._branch_order]
        resolved = sum(branch.status == "success" for branch in branches)
        completed_branches = sum(
            branch.status in {"success", "error"} for branch in branches
        )
        llm_tasks = [task for task in self.tasks.values() if task.kind == "llm"]
        deterministic_tasks = [
            task for task in self.tasks.values() if task.kind == "deterministic"
        ]
        retries = sum(
            task.node_name == "repair_invalid_extraction"
            for task in self.tasks.values()
        )

        title = (
            f"  {self._style('CIPOC', _Color.BOLD, _Color.MAIN)}"
            f"{self._style(' / ' + self.description.upper(), _Color.GREY)}"
        )
        target_stats = f"{resolved}/{len(branches)} variables resolved" if branches else ""
        if final:
            llm_word = "call" if len(llm_tasks) == 1 else "calls"
            step_word = "step" if len(deterministic_tasks) == 1 else "steps"
            work_stats = (
                f"{self._style('✓', _Color.SUCCESS)} {len(llm_tasks)} LLM {llm_word}"
                f"   {self._style('◆', _Color.MAIN)} {len(deterministic_tasks)} deterministic {step_word}"
            )
        else:
            active_llm = sum(task.status == "active" for task in llm_tasks)
            active_deterministic = sum(
                task.status == "active" for task in deterministic_tasks
            )
            work_stats = (
                f"{self._style('◉', _Color.LLM)} {active_llm} LLM active"
                f"   {self._style('◆', _Color.MAIN)} {active_deterministic} deterministic active"
            )
        retry_word = "retry" if retries == 1 else "retries"
        retry_stats = f"   {self._style('↻', _Color.LLM)} {retries} {retry_word}"

        lines = [
            "",
            f"{title}{' ' * 4}{self._style(target_stats, _Color.WHITE)}",
            f"  {work_stats}{retry_stats}   {elapsed:0.1f}s",
            f"  [{self._style(_progress_bar(completed_branches, len(branches)), _Color.MAIN)}]"
            f"  {completed_branches}/{len(branches)} branches complete",
            "",
            f"  {self._style('SHARED WORK', _Color.BOLD, _Color.GREY)}",
        ]

        shared_tasks = [
            self.tasks[task_id]
            for task_id in self._order
            if self.tasks[task_id].branch_key is None
            and self.tasks[task_id].kind != "container"
        ]
        if not shared_tasks:
            lines.append(f"  {self._style('○ Waiting to start', _Color.DIM)}")
        for task in shared_tasks:
            timing = (
                self._style(self._duration(task), _Color.DIM)
                if task.status != "active"
                else self._style(f"running {self._duration(task)}", _Color.ACTIVE)
            )
            lines.append(
                f"  {self._status_symbol(task)} {self._style(_humanize(task.node_name), _Color.WHITE)}"
                f" {self._kind_symbol(task)}  {timing}"
            )

        lines.extend(["", f"  {self._style('PARALLEL BRANCHES', _Color.BOLD, _Color.GREY)}"])
        if not branches:
            lines.append(f"  {self._style('○ No branches discovered', _Color.DIM)}")

        visible_branches = branches[:10]
        for index, branch in enumerate(visible_branches):
            last = index == len(visible_branches) - 1
            head = "└" if last else "├"
            stem = " " if last else "│"
            branch_title = _truncate(f"{branch.item_id}  {branch.name}", width - 8)
            lines.append(f"  {head} {self._style(branch_title, _Color.BOLD, _Color.WHITE)}")

            branch_tasks = [self.tasks[task_id] for task_id in branch.task_ids]
            if not branch_tasks:
                state = "starting…" if branch.status == "active" else "queued"
                lines.append(f"  {stem} {self._style('○ ' + state, _Color.DIM)}")
            for task in branch_tasks:
                lines.extend(self._operation_line(task, stem, width))

            if branch.status == "success":
                value = branch.value or "No value"
                lines.append(
                    f"  {stem} {self._style('✓ ' + _truncate(value, width - 22), _Color.SUCCESS)}"
                    f"  {self._style('complete', _Color.DIM)}"
                )
            elif branch.status == "error":
                lines.append(f"  {stem} {self._style('✗ unresolved', _Color.ERROR)}")
                if branch.errors:
                    lines.append(
                        f"  {stem}   {self._style('↳ ' + _truncate(branch.errors[0], width - 10), _Color.ERROR)}"
                    )
            lines.append(f"  {stem}")

        if len(branches) > len(visible_branches):
            lines.append(
                f"  {self._style(f'… {len(branches) - len(visible_branches)} more branches', _Color.DIM)}"
            )

        if self._activities:
            lines.append(f"  {self._style('RECENT ACTIVITY', _Color.BOLD, _Color.GREY)}")
            for timestamp, text, status in self._activities[-4:]:
                color = (
                    _Color.SUCCESS
                    if status == "success"
                    else _Color.ERROR
                    if status == "error"
                    else _Color.LLM
                )
                lines.append(
                    f"  {self._style(f'{timestamp:06.1f}', _Color.DIM)}  "
                    f"{self._style(_truncate(text, width - 12), color)}"
                )

        if self._fatal_error:
            lines.extend(
                [
                    "",
                    f"  {self._style('✗ Run failed', _Color.BOLD, _Color.ERROR)}"
                    f"  {self._style(_truncate(self._fatal_error, width - 20), _Color.ERROR)}",
                ]
            )
        elif final:
            lines.extend(
                [
                    "",
                    f"  {self._style('✓ Extraction complete', _Color.BOLD, _Color.SUCCESS)}"
                    f"{' ' * 4}{elapsed:0.1f}s",
                ]
            )
        return lines


def run_with_progress(
    graph: CompiledStateGraph,
    graph_input: Any,
    *,
    subgraphs: bool = False,
    description: str = "Agent",
    node_kinds: Mapping[str, TaskKind] | None = None,
    show_branches: bool = False,
) -> Any:
    """Run a LangGraph graph with live progress and return its final state.

    Set ``subgraphs=True`` to include nodes inside compiled subgraphs. Known
    CIPOC nodes are classified automatically. Pass ``node_kinds`` to
    override or extend those classifications for custom graphs. The branch board
    additionally groups subgraph tasks by variable metadata in their input.
    """
    final_result: Any = None
    display: _ProgressDisplay
    if show_branches:
        display = _BranchProgressDisplay(
            description,
            graph_input,
            node_kinds=node_kinds,
        )
    else:
        display = _ProgressDisplay(
            description,
            node_kinds=node_kinds,
            expected_steps=_expected_step_count(graph, node_kinds),
        )
    display.draw()

    try:
        stream = graph.stream(
            graph_input,
            stream_mode=["tasks", "values"],
            subgraphs=subgraphs,
        )

        for item in stream:
            if subgraphs:
                namespace, mode, event = item
            else:
                mode, event = item
                namespace = ()

            if mode == "values":
                if not namespace:
                    final_result = event
                continue

            task_id = str(event["id"])
            node_name = event["name"]
            if "input" in event:
                display.start(task_id, namespace, node_name, event["input"])
            else:
                display.finish(
                    task_id,
                    namespace,
                    node_name,
                    event.get("result"),
                    event.get("error"),
                )
    except BaseException as error:
        display.fail(error)
        raise

    if final_result is None:
        error = RuntimeError("Graph produced no final state.")
        display.fail(error)
        raise error

    display.complete()
    return final_result


__all__ = ["run_with_progress"]

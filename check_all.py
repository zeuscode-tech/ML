#!/usr/bin/env python3
"""Безопасная проверка структуры и исходных файлов всех работ по ML."""

from __future__ import annotations

import ast
import json
import re
import urllib.parse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PRACTICES = [ROOT / f"practice-{number}" for number in range(1, 14)]


def check_structure() -> None:
    for number, directory in enumerate(PRACTICES, start=1):
        assert directory.is_dir(), f"Нет папки practice-{number}"
        assert (directory / "README.md").is_file(), f"Нет practice-{number}/README.md"
        assert (directory / "requirements.txt").is_file(), (
            f"Нет practice-{number}/requirements.txt"
        )
        implementations = list(directory.rglob("*.py")) + list(directory.rglob("*.ipynb"))
        assert implementations, f"В practice-{number} нет кода или ноутбука"


def check_markdown_links() -> None:
    pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    broken: list[str] = []
    for readme in sorted(ROOT.rglob("*.md")):
        if any(part in {".git", ".venv"} for part in readme.parts):
            continue
        for target in pattern.findall(readme.read_text(encoding="utf-8")):
            target = target.strip().split()[0].strip("<>")
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            relative = urllib.parse.unquote(target.split("#", 1)[0])
            if relative and not (readme.parent / relative).exists():
                broken.append(f"{readme.relative_to(ROOT)} -> {target}")
    assert not broken, "Битые ссылки:\n" + "\n".join(broken)


def check_python_syntax() -> None:
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def check_notebooks() -> None:
    notebooks = sorted({path for directory in PRACTICES for path in directory.rglob("*.ipynb")})
    assert notebooks, "Ноутбуки не найдены"
    failures: list[str] = []
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook.get("nbformat") == 4, f"Неожиданный формат: {path}"
        code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
        if not code_cells:
            failures.append(f"{path.relative_to(ROOT)}: нет ячеек с кодом")
            continue
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        if errors:
            failures.append(f"{path.relative_to(ROOT)}: сохранены ошибки выполнения")
        if not all(cell.get("execution_count") is not None for cell in code_cells):
            failures.append(f"{path.relative_to(ROOT)}: не все ячейки выполнены")
    assert not failures, "Проблемы ноутбуков:\n" + "\n".join(failures)


def main() -> None:
    checks = [
        ("структура practice-1…practice-13", check_structure),
        ("локальные ссылки README", check_markdown_links),
        ("синтаксис Python", check_python_syntax),
        ("Jupyter Notebook", check_notebooks),
    ]
    for title, function in checks:
        function()
        print(f"[OK] {title}")
    print("Все 13 практических работ прошли автоматическую проверку.")


if __name__ == "__main__":
    main()

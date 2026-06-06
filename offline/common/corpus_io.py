import json
from collections.abc import Iterable, Iterator
from pathlib import Path


def write_rows(path: str | Path, rows: Iterable[dict], append: bool = False) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with out_path.open(mode) as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_rollouts(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def merge_corpus(paths: Iterable[str | Path], out_path: str | Path) -> None:
    def rows() -> Iterator[dict]:
        for path in paths:
            yield from read_rollouts(path)

    write_rows(out_path, rows())

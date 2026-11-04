import os
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO

import anyio


class LocalStorage:
    """Stores files on the local filesystem under a single root directory.

    Writes go to a temporary file in the target directory and are then
    renamed into place, so a crash never leaves a half-written file under a
    real key.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _path_for(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError(f"Invalid storage key: {key!r}")
        return path

    async def save(self, key: str, source: BinaryIO) -> None:
        await anyio.to_thread.run_sync(self._save_sync, self._path_for(key), source)

    @staticmethod
    def _save_sync(path: Path, source: BinaryIO) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        source.seek(0)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".upload-")
        try:
            with os.fdopen(fd, "wb") as tmp:
                shutil.copyfileobj(source, tmp)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    async def read(self, key: str) -> bytes:
        path = self._path_for(key)
        return await anyio.to_thread.run_sync(path.read_bytes)

    async def delete(self, key: str) -> None:
        path = self._path_for(key)
        await anyio.to_thread.run_sync(lambda: path.unlink(missing_ok=True))

    async def exists(self, key: str) -> bool:
        path = self._path_for(key)
        return await anyio.to_thread.run_sync(path.is_file)

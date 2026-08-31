"""Download and validate the image fonts used by the local renderer."""
from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import aiohttp


FONT_DIR = Path(__file__).resolve().parent.parent / "resource" / "font"
FONT_RELEASE_BASE_URL = (
    "https://github.com/drdon1234/fonts/releases/download/v1.0.0"
)
_DOWNLOAD_CHUNK_SIZE = 256 * 1024
_DOWNLOAD_TIMEOUT_SECONDS = 600
_USER_AGENT = "astrbot_plugin_ai_summary/0.4.2"


@dataclass(frozen=True)
class FontAsset:
    """One immutable font release asset."""

    filename: str
    size: int
    sha256: str


FONT_ASSETS = (
    FontAsset(
        "LXGWWenKai-Medium.ttf",
        25_379_848,
        "d4bdeb38a39151d74d084cba5090f8cb7d20bf83eedb78c35939ae70b9f4e3f6",
    ),
    FontAsset(
        "LXGWWenKai-Regular.ttf",
        25_575_676,
        "39ad71264b588165b469e35e6afb162a378dacd1f95348160240ba9038ac3009",
    ),
    FontAsset(
        "NotoSansCJKsc-Bold.otf",
        17_002_248,
        "b5f0d1a190a7f9b43c310a8850630af12553df32c4c050543f9059732d9b4c0a",
    ),
    FontAsset(
        "NotoSansCJKsc-Regular.otf",
        16_437_364,
        "2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b",
    ),
    FontAsset(
        "NotoSerifCJKsc-Bold.otf",
        25_521_460,
        "8af07d4b6c2e82bcc72a30e066eaf295f11b9424f4aad2eaa9fe0e9c3b38fc73",
    ),
    FontAsset(
        "NotoSerifCJKsc-Regular.otf",
        24_543_080,
        "2a2eae2628df83556c54018c41e20fa532c1b862c5256ae8b3f23feb918d12ca",
    ),
    FontAsset(
        "ZCOOLQingKeHuangYou-Regular.ttf",
        8_328_684,
        "54f0c0df4308cd74cd0f2fd3494ae054dbc4a1fd6fa7d71f4807eb4cdd8b4136",
    ),
    FontAsset(
        "ZCOOLXiaoWei-Regular.ttf",
        6_313_808,
        "a42b620140f493db42f741351dfbf343c0936d58588ee8004b8b2a218d997ff1",
    ),
)


class FontAssetError(RuntimeError):
    """Raised when a required font cannot be downloaded or validated."""


class FontAssetManager:
    """Serialize font preparation and keep verified release assets on disk."""

    def __init__(
        self,
        font_dir: Path = FONT_DIR,
        assets: Sequence[FontAsset] = FONT_ASSETS,
        base_url: str = FONT_RELEASE_BASE_URL,
        timeout_seconds: int = _DOWNLOAD_TIMEOUT_SECONDS,
    ) -> None:
        self.font_dir = Path(font_dir)
        self.assets = tuple(assets)
        self.base_url = str(base_url).rstrip("/")
        self.timeout_seconds = max(10, int(timeout_seconds))
        self._lock = asyncio.Lock()
        self._ready = False

    async def ensure_ready(self) -> None:
        """Ensure every configured font exists and matches the release manifest."""
        if self._ready and self._expected_sizes_exist():
            return

        async with self._lock:
            if self._ready and self._expected_sizes_exist():
                return

            self._ready = False
            try:
                self.font_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise FontAssetError(
                    f"无法创建字体目录 {self.font_dir}: {exc}"
                ) from exc
            missing_or_invalid = []
            for asset in self.assets:
                if not await asyncio.to_thread(self._asset_matches, asset):
                    missing_or_invalid.append(asset)

            if missing_or_invalid:
                timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
                try:
                    async with aiohttp.ClientSession(
                        timeout=timeout,
                        headers={"User-Agent": _USER_AGENT},
                    ) as session:
                        for asset in missing_or_invalid:
                            await self._download_asset(session, asset)
                except asyncio.CancelledError:
                    raise
                except FontAssetError:
                    raise
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                    raise FontAssetError(
                        f"字体资源下载会话失败: {exc}"
                    ) from exc

            for asset in self.assets:
                if not await asyncio.to_thread(self._asset_matches, asset):
                    raise FontAssetError(f"字体文件校验失败: {asset.filename}")
            self._ready = True

    def _expected_sizes_exist(self) -> bool:
        """Cheaply recheck the ready cache without hashing every font again."""
        for asset in self.assets:
            path = self.font_dir / asset.filename
            try:
                if not path.is_file() or path.stat().st_size != asset.size:
                    return False
            except OSError:
                return False
        return True

    def _asset_matches(self, asset: FontAsset) -> bool:
        path = self.font_dir / asset.filename
        try:
            if not path.is_file() or path.stat().st_size != asset.size:
                return False
            digest = hashlib.sha256()
            with path.open("rb") as file_obj:
                while chunk := file_obj.read(1024 * 1024):
                    digest.update(chunk)
            return digest.hexdigest() == asset.sha256
        except OSError:
            return False

    async def _download_asset(
        self,
        session: aiohttp.ClientSession,
        asset: FontAsset,
    ) -> None:
        target_path = self.font_dir / asset.filename
        temp_path = self.font_dir / (
            f".{asset.filename}.{uuid.uuid4().hex}.part"
        )
        received = 0
        digest = hashlib.sha256()
        try:
            url = f"{self.base_url}/{asset.filename}"
            async with session.get(url, allow_redirects=True) as response:
                response.raise_for_status()
                with temp_path.open("xb") as file_obj:
                    async for chunk in response.content.iter_chunked(
                        _DOWNLOAD_CHUNK_SIZE
                    ):
                        if not chunk:
                            continue
                        received += len(chunk)
                        if received > asset.size:
                            raise FontAssetError(
                                f"字体文件大小超出预期: {asset.filename}"
                            )
                        digest.update(chunk)
                        file_obj.write(chunk)

            if received != asset.size:
                raise FontAssetError(
                    f"字体文件大小不匹配: {asset.filename} "
                    f"({received} != {asset.size})"
                )
            if digest.hexdigest() != asset.sha256:
                raise FontAssetError(f"字体文件哈希不匹配: {asset.filename}")
            temp_path.replace(target_path)
        except asyncio.CancelledError:
            raise
        except FontAssetError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            raise FontAssetError(
                f"字体文件下载失败: {asset.filename}: {exc}"
            ) from exc
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


FONT_ASSET_MANAGER = FontAssetManager()


async def ensure_font_assets() -> None:
    """Ensure the shared renderer font directory is complete."""
    await FONT_ASSET_MANAGER.ensure_ready()

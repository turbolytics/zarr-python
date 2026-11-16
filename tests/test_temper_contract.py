"""The store.contract and array.contract lattices (temper/lattices).

Each test states what it proves with @pytest.mark.covers or
@pytest.mark.exempts (temper-python). Without `temper run`, the marks do
nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numcodecs
import numpy as np
import pytest

import zarr
import zarr.codecs as zc
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import (
    FsspecStore,
    LocalStore,
    LoggingStore,
    MemoryStore,
    ObjectStore,
    WrapperStore,
    ZipStore,
)

if TYPE_CHECKING:
    from pathlib import Path

    from zarr.abc.store import Store

pytestmark = [
    # LoggingStore logs every call. The zip archive warns when a key is rewritten.
    pytest.mark.filterwarnings("ignore::UserWarning"),
    # zarr warns, by design, that fixed-length strings have no v3 specification yet.
    pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning"),
]

proto = default_buffer_prototype()
STORES = ["memory", "local", "zip", "fsspec", "obstore", "logging"]
FORMATS = [2, 3]


def make_store(kind: str, tmp_path: Path) -> Store:
    if kind == "memory":
        return MemoryStore()
    if kind == "local":
        return LocalStore(tmp_path / "local")
    if kind == "zip":
        return ZipStore(tmp_path / "store.zip", mode="w")
    if kind == "fsspec":
        fsspec = pytest.importorskip("fsspec")
        from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

        fs = AsyncFileSystemWrapper(fsspec.filesystem("file", auto_mkdir=True), asynchronous=True)
        return FsspecStore(fs=fs, path=str(tmp_path / "fsspec"))
    if kind == "obstore":
        obstore = pytest.importorskip("obstore.store")
        return ObjectStore(obstore.MemoryStore())
    if kind == "logging":
        return LoggingStore(MemoryStore())
    raise ValueError(kind)


def store_param(member: str, exclude: tuple[str, ...] = ()) -> list[Any]:
    return [
        pytest.param(
            kind,
            fmt,
            id=f"{kind}-v{fmt}",
            marks=[pytest.mark.covers(f"store.contract/{member}", subject=f"store.{kind}")]
            # The round trip also proves the store works at all.
            + ([pytest.mark.covers(f"store.{kind}")] if member == "roundtrip" else []),
        )
        for kind in STORES
        if kind not in exclude
        for fmt in FORMATS
    ]


def format_param(member: str) -> list[Any]:
    return [
        pytest.param(
            fmt,
            id=f"v{fmt}",
            marks=[pytest.mark.covers(f"array.contract/{member}", subject=f"format.v{fmt}")]
            + ([pytest.mark.covers(f"format.v{fmt}")] if member == "codec_roundtrip" else []),
        )
        for fmt in FORMATS
    ]


def sibling_group(store: Store, fmt: int) -> zarr.Group:
    """A group holding lat and lat_bnds, the CF naming pattern."""
    group = zarr.open_group(store, mode="w", zarr_format=fmt)
    lat = group.create_array("lat", shape=(4,), chunks=(2,), dtype="f8")
    bnds = group.create_array("lat_bnds", shape=(4, 2), chunks=(2, 2), dtype="f8")
    lat[:2] = 1.0
    bnds[:] = 2.0
    return group


# store.contract


@pytest.mark.parametrize(("kind", "fmt"), store_param("roundtrip"))
def test_store_roundtrip(kind: str, fmt: int, tmp_path: Path) -> None:
    store = make_store(kind, tmp_path)
    data = np.arange(60, dtype="i4").reshape(6, 10)
    kwargs: dict[str, Any] = {"shards": (4, 6)} if fmt == 3 else {}
    a = zarr.create_array(store, shape=data.shape, chunks=(2, 3), dtype="i4", zarr_format=fmt, **kwargs)
    a[:] = data
    # ZipStore cannot make a read-only copy, so reopen in the default mode.
    np.testing.assert_array_equal(zarr.open_array(store)[:], data)


@pytest.mark.parametrize(("kind", "fmt"), store_param("chunk_count_with_siblings"))
def test_chunk_count_with_siblings(kind: str, fmt: int, tmp_path: Path) -> None:
    lat = sibling_group(make_store(kind, tmp_path), fmt)["lat"]
    assert lat.nchunks_initialized == 1
    lat.info_complete()


# ZipStore cannot delete. test_zip_refuses_deletes proves its exemption.
@pytest.mark.parametrize(("kind", "fmt"), store_param("delete_keeps_siblings", exclude=("zip",)))
def test_delete_keeps_siblings(kind: str, fmt: int, tmp_path: Path) -> None:
    store = make_store(kind, tmp_path)
    group = sibling_group(store, fmt)
    del group["lat"]
    reopened = zarr.open_group(store, mode="r", zarr_format=fmt)
    assert "lat" not in reopened
    np.testing.assert_array_equal(reopened["lat_bnds"][:], np.full((4, 2), 2.0))


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param(
            "zip",
            marks=pytest.mark.exempts("store.contract/delete_keeps_siblings", subject="store.zip"),
        )
    ],
)
def test_zip_refuses_deletes(kind: str, tmp_path: Path) -> None:
    group = sibling_group(make_store(kind, tmp_path), 3)
    with pytest.raises(NotImplementedError):
        del group["lat"]


class RecordingStore(WrapperStore[Any]):
    """Records every call that would change the store."""

    writes: list[tuple[str, str]]

    async def set(self, key: str, value: Any) -> None:
        self.writes.append(("set", key))
        await self._store.set(key, value)

    async def set_if_not_exists(self, key: str, value: Any) -> None:
        self.writes.append(("set_if_not_exists", key))
        await self._store.set_if_not_exists(key, value)

    async def delete(self, key: str) -> None:
        self.writes.append(("delete", key))
        await self._store.delete(key)


@pytest.mark.parametrize(("kind", "fmt"), store_param("read_never_writes"))
def test_read_never_writes(kind: str, fmt: int, tmp_path: Path) -> None:
    store = make_store(kind, tmp_path)
    group = zarr.open_group(store, mode="w", zarr_format=fmt)
    group.create_array("x", shape=(6,), chunks=(2,), dtype="f8")[:4] = 1.5
    recording = RecordingStore(store)
    recording.writes = []
    reopened = zarr.open_group(recording)
    reopened["x"][:]
    _ = dict(reopened.members())
    assert recording.writes == []


@pytest.mark.parametrize(("kind", "fmt"), store_param("byte_range_edges"))
async def test_byte_range_edges(kind: str, fmt: int, tmp_path: Path) -> None:
    store = make_store(kind, tmp_path)
    await store._open()
    await store.set("k", proto.buffer.from_bytes(b"0123456789"))
    for request in (
        RangeByteRequest(4, 4),
        OffsetByteRequest(10),
        OffsetByteRequest(15),
        SuffixByteRequest(0),
    ):
        got = await store.get("k", proto, byte_range=request)
        assert got is not None
        assert got.to_bytes() == b"", request


# array.contract


def v3_pipelines() -> dict[str, dict[str, Any]]:
    return {
        "default": {},
        "big-endian bytes": {"serializer": zc.BytesCodec(endian="big")},
        "gzip": {"compressors": [zc.GzipCodec()]},
        "blosc bitshuffle": {"compressors": [zc.BloscCodec(cname="zstd", shuffle="bitshuffle")]},
        "transpose": {"filters": [zc.TransposeCodec(order=(1, 0))]},
        "sharding": {"shards": (4, 6)},
        "sharding+transpose": {"shards": (4, 6), "filters": [zc.TransposeCodec(order=(1, 0))]},
    }


def v2_pipelines() -> dict[str, dict[str, Any]]:
    return {
        "default": {},
        "no compressor": {"compressors": None},
        "zlib": {"compressors": numcodecs.Zlib()},
        "blosc autoshuffle": {"compressors": numcodecs.Blosc(cname="lz4", shuffle=numcodecs.Blosc.AUTOSHUFFLE)},
        "order F": {"order": "F"},
    }


DTYPES = ["bool", "i1", "u8", "i8", "f2", "f8", "c16", "M8[ns]", "U5", "S5"]


def sample(dtype: str) -> np.ndarray:
    rng = np.random.default_rng(0)
    dt = np.dtype(dtype)
    n = 35
    if dt.kind == "b":
        a = rng.integers(0, 2, n).astype(dt)
    elif dt.kind in "iu":
        info = np.iinfo(dt)
        a = rng.integers(info.min, info.max, n, dtype=dt, endpoint=True)
    elif dt.kind == "f":
        a = rng.standard_normal(n).astype(dt)
        a[:4] = [np.nan, np.inf, -np.inf, -0.0]
    elif dt.kind == "c":
        a = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(dt)
    elif dt.kind == "M":
        a = rng.integers(-(10**9), 10**9, n).astype(dt)
        a[0] = np.datetime64("NaT", "ns")
    elif dt.kind == "U":
        a = np.array([["", "a", "é", "ünï", "abcde"][i % 5] for i in range(n)], dtype=dt)
    else:
        a = np.array([[b"", b"a", b"\x00b", b"abcde", b"\xff"][i % 5] for i in range(n)], dtype=dt)
    return a.reshape(5, 7)


def same(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype.kind in "Mm":
        return np.array_equal(a.view("i8"), b.view("i8"))
    if a.dtype.kind in "fc":
        return np.array_equal(a, b, equal_nan=True) and np.array_equal(
            np.signbit(a.real), np.signbit(b.real)
        )
    return np.array_equal(a, b)


@pytest.mark.parametrize("fmt", format_param("codec_roundtrip"))
def test_codec_roundtrip(fmt: int) -> None:
    pipelines = v3_pipelines() if fmt == 3 else v2_pipelines()
    for name, kwargs in pipelines.items():
        for dtype in DTYPES:
            if name == "big-endian bytes" and np.dtype(dtype).kind in "US":
                continue  # string dtypes require their own serializer; creation refuses
            store = MemoryStore()
            data = sample(dtype)
            a = zarr.create_array(store, shape=data.shape, chunks=(2, 3), dtype=dtype, zarr_format=fmt, **kwargs)
            a[:] = data
            back = zarr.open_array(store, mode="r")[:]
            assert same(back, data), (name, dtype)


FILLS = [
    ("f8", np.nan), ("f8", np.inf), ("f8", -0.0), ("f2", np.nan), ("c16", complex(np.nan, 0)),
    ("i8", np.iinfo("i8").min), ("u8", np.iinfo("u8").max), ("bool", True),
    ("M8[ns]", "NaT"), ("m8[s]", -5), ("U5", "é"), ("S5", b"\x00a"),
]


@pytest.mark.parametrize("fmt", format_param("fill_values"))
def test_fill_values(fmt: int) -> None:
    for dtype, fill in FILLS:
        value = np.array(fill).astype(dtype)[()] if dtype[0] in "Mm" else np.array(fill, dtype=dtype)[()]
        store = MemoryStore()
        zarr.create_array(store, shape=(3,), chunks=(2,), dtype=dtype, zarr_format=fmt, fill_value=value)
        a = zarr.open_array(store, mode="r")
        want = np.array(value, dtype=a.dtype).tobytes()
        assert np.array(a.fill_value, dtype=a.dtype).tobytes() == want, (dtype, fill)
        assert a[:][:1].tobytes() == want, (dtype, fill)


@pytest.mark.parametrize("fmt", format_param("indexing"))
def test_indexing(fmt: int) -> None:
    ref = np.random.default_rng(1).integers(0, 1000, (9, 11)).astype("i4")
    order_f: dict[str, Any] = {"order": "F"} if fmt == 2 else {"config": {"order": "F"}}
    layouts: list[dict[str, Any]] = [{}, order_f] + ([{"shards": (6, 6)}] if fmt == 3 else [])
    for kwargs in layouts:
        a = zarr.create_array(MemoryStore(), shape=ref.shape, chunks=(2, 3), dtype="i4", zarr_format=fmt, **kwargs)
        a[:] = ref
        np.testing.assert_array_equal(a[1:8:3, ::2], ref[1:8:3, ::2])
        np.testing.assert_array_equal(a[-3:, :-4], ref[-3:, :-4])
        np.testing.assert_array_equal(a[..., 3], ref[..., 3])
        np.testing.assert_array_equal(a.oindex[[8, 0, 8], :], ref[[8, 0, 8], :])
        np.testing.assert_array_equal(a.vindex[[-1, 0], [-1, 0]], ref[[-1, 0], [-1, 0]])
        mask = ref % 7 == 0
        np.testing.assert_array_equal(a.vindex[mask], ref[mask])


@pytest.mark.parametrize("fmt", format_param("checksum_detects_corruption"))
def test_checksum_detects_corruption(fmt: int) -> None:
    kwargs: dict[str, Any] = (
        {"compressors": [zc.Crc32cCodec()]} if fmt == 3 else {"compressors": numcodecs.CRC32()}
    )
    for damage in ("truncate", "flip"):
        store = MemoryStore()
        a = zarr.create_array(store, shape=(8,), chunks=(2,), dtype="i8", fill_value=0, zarr_format=fmt, **kwargs)
        a[:] = np.arange(1, 9)
        key = "c/0" if fmt == 3 else "0"
        raw = bytearray(store._store_dict[key].to_bytes())
        if damage == "truncate":
            raw = raw[:-3]
        else:
            raw[len(raw) // 2] ^= 0xFF
        store._store_dict[key] = proto.buffer.from_bytes(bytes(raw))
        with pytest.raises(Exception):  # noqa: B017, PT011
            zarr.open_array(store, mode="r")[:]


@pytest.mark.parametrize(
    "fmt",
    [
        pytest.param(
            fmt,
            id=f"v{fmt}",
            marks=pytest.mark.exempts("array.contract/resize_no_resurrection", subject=f"format.v{fmt}"),
        )
        for fmt in FORMATS
    ],
)
def test_resize_restores_boundary_data(fmt: int) -> None:
    """Array.resize documents that growing restores data kept in boundary chunks."""
    a = zarr.create_array(MemoryStore(), shape=(8,), chunks=(2,), dtype="i4", fill_value=0, zarr_format=fmt)
    a[:] = np.arange(1, 9)
    a.resize((3,))
    a.resize((8,))
    np.testing.assert_array_equal(a[:], [1, 2, 3, 4, 0, 0, 0, 0])

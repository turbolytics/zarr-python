# zarr-python contract map

[Temper](https://github.com/turbolytics/temper) map of what zarr's stores and
arrays promise. `MATRIX.md` at the repository root is the result.

- `lattices/store.contract.yml`: what every store must do, checked on six stores
  in both zarr formats.
- `lattices/array.contract.yml`: what an array promises in each zarr format.
- `subjects/`: one file per store or format.
- `../tests/test_temper_contract.py`: the tests. Each one states what it proves
  with `@pytest.mark.covers`.

Run it:

```sh
pip install -e . --group test fsspec obstore temper-python
brew install turbolytics/tap/temper   # or: go install github.com/turbolytics/temper/cmd/temper@v0.1.2
temper clean
temper run --level ci -- pytest tests/test_temper_contract.py
temper matrix
```

`temper matrix` exits 1 while any cell is failing.

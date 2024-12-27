"""Bagz file reader/writer and PyGrain-compatible data source for POSIX systems.

Bagz is a file format for storing a sequence of string records, typically
serialised protocol buffers. It supports fast index based look-up.
"""

import bisect
from collections.abc import Sequence
import itertools
import mmap
import os
import re
import shutil
import struct
import tqdm

from searchless_chess.src import constants
from datasets import Dataset, concatenate_datasets
from typing import Any, SupportsIndex
from typing_extensions import Self
import zstandard as zstd


class BagFileReader(Sequence[bytes]):
  """Reader for single Bagz files."""

  def __init__(
      self,
      filename: str,
      *,
      separate_limits: bool = False,
      decompress: bool | None = None,
  ) -> None:
    """Creates a BagFileReader.

    Args:
      filename: The name of the single Bagz file to read.
      separate_limits: Whether the limits are stored in a separate file.
      decompress: Whether to decompress the records. If None, uses the file
        extension to determine whether to decompress.
    """
    if decompress or (decompress is None and filename.endswith('.bagz')):
      self._process = lambda x: zstd.decompress(x) if x else x
    else:
      self._process = lambda x: x
    self._filename = filename
    fd = os.open(filename, os.O_RDONLY)
    try:
      self._records = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
      file_size = self._records.size()
    except ValueError:
      self._records = b''
      file_size = 0
    finally:
      os.close(fd)
    if separate_limits:
      directory, name = os.path.split(filename)
      fd = os.open(os.path.join(directory, 'limits.' + name), os.O_RDONLY)
      try:
        self._limits = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        index_size = self._limits.size()
      except ValueError:
        self._limits = b''
        index_size = 0
      finally:
        os.close(fd)
      index_start = 0
    else:
      if 0 < file_size < 8:
        raise ValueError('Bagz file too small')
      self._limits = self._records
      if file_size:
        (index_start,) = struct.unpack('<Q', self._records[-8:])
      else:
        index_start = 0
      assert file_size >= index_start
      index_size = file_size - index_start
    assert index_size % 8 == 0
    self._num_records = index_size // 8
    self._limits_start = index_start

  def __len__(self) -> int:
    """Returns the number of records in the Bagz file."""
    return self._num_records

  def __getitem__(self, index: SupportsIndex) -> bytes:
    """Returns a record from the Bagz file."""
    i = index.__index__()
    if not 0 <= i < self._num_records:
      raise IndexError('bagz.BragReader index out of range')
    end = i * 8 + self._limits_start
    if i:
      rec_range = struct.unpack('<2q', self._limits[end - 8 : end + 8])
    else:
      rec_range = (0, *struct.unpack('<q', self._limits[end : end + 8]))
    return self._process(self._records[slice(*rec_range)])


class BagShardReader(Sequence[bytes]):
  """Reader for sharded Bagz files."""

  def __init__(
      self,
      filename: str,
      *,
      separate_limits: bool = False,
      decompress: bool | None = None,
  ) -> None:
    """Creates a BagShardReader.

    Args:
      filename: The name of the sharded Bagz file to read.
      separate_limits: Whether the limits are stored in a separate file.
      decompress: Whether to decompress the records. If None, uses the file
        extension to determine whether to decompress.
    """
    matches = re.findall(r'@(\d+)', filename)
    assert len(matches) == 1
    num_files = int(matches[0])
    assert num_files < 100_000
    self._bags = tuple(
        BagFileReader(
            filename=re.sub(
                r'@(\d+)', f'-{idx:05d}-of-{num_files:05d}', filename
            ),
            separate_limits=separate_limits,
            decompress=decompress,
        )
        for idx in range(num_files)
    )
    self._accum = tuple(itertools.accumulate(map(len, self._bags)))

  def __len__(self) -> int:
    """Returns the number of records in the Bagz file."""
    return self._accum[-1]

  def __getitem__(self, index: int) -> bytes:
    if index < 0:
      index += self._accum[-1]
    if seqn := bisect.bisect_left(self._accum, index + 1):
      index -= self._accum[seqn - 1]
    return self._bags[seqn][index]


class BagReader(Sequence[bytes]):
  """Reader for Bagz files."""

  def __init__(
      self,
      filename: str,
      *,
      separate_limits: bool = False,
      decompress: bool | None = None,
  ) -> None:
    """Creates a BagReader.

    Args:
      filename: The name of the Bagz file to read. Supports the @N shard syntax
        (where @0 corresponds to the single file case). If the shard syntax does
        not parse, then `filename` is treated as a single file.
      separate_limits: Whether the limits are stored in a separate file.
      decompress: Whether to decompress the records. If None, uses the file
        extension to determine whether to decompress.
    """
    if matches := re.findall(r'@(\d+)', filename):
      assert len(matches) == 1
      if int(matches[0]) != '0':
        reader_class = BagShardReader
      else:
        filename = filename.replace(matches[0], '')
        reader_class = BagFileReader
    else:
      reader_class = BagFileReader

    self._reader = reader_class(
        filename=filename,
        separate_limits=separate_limits,
        decompress=decompress,
    )

  def __len__(self) -> int:
    """Returns the number of records in the Bagz file."""
    return len(self._reader)

  def __getitem__(self, index: SupportsIndex) -> bytes:
    """Returns a record from the Bagz file."""
    return self._reader[index]


class BagWriter:
  """Writer for Bagz files."""

  def __init__(
      self,
      filename: str,
      *,
      separate_limits: bool = False,
      compress: bool | None = None,
      compression_level: int = 0,
  ) -> None:
    """Creates a BagWriter.

    Args:
      filename: The name of the Bagz file to write.
      separate_limits: Whether to keep the limits in a separate file.
      compress: Whether to compress the records. If None, uses the file
        extension to determine whether to compress.
      compression_level: The compression level to use when compressing.
    """
    if compress or (compress is None and filename.endswith('.bagz')):
      self._process = zstd.ZstdCompressor(level=compression_level).compress
    else:
      self._process = lambda x: x
    self._separate_limits = separate_limits
    directory, name = os.path.split(filename)
    self._records = open(filename, 'wb')
    self._limits = open(os.path.join(directory, 'limits.' + name), 'wb+')

  def write(self, data: bytes) -> None:
    """Writes a record to the Bagz file."""
    if data:
      self._records.write(self._process(data))
    self._limits.write(struct.pack('<q', self._records.tell()))

  def flush(self) -> None:
    """Flushes the Bagz file."""
    self._records.flush()
    self._limits.flush()

  def __enter__(self) -> Self:
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    """Ensures the Bagz file is closed when exiting a context."""
    self.close()

  def close(self) -> None:
    """Concatenates the limits file to the end of the data file."""
    if self._separate_limits:
      self._records.close()
      self._limits.close()
    else:
      self._limits.seek(0)
      shutil.copyfileobj(self._limits, self._records)
      self._records.close()
      os.unlink(self._limits.name)
      self._limits.close()


class BagDataSource:
  """PyGrain-compatible data source for bagz files."""

  def __init__(self, path: str) -> None:
    """Creates a new BagDataSource object.

    Args:
      path: The path to the bag file.
    """
    self._path = os.fspath(path)
    self._reader = BagReader(self._path)
    self._num_records = len(self._reader)

  def __len__(self) -> int:
    return self._num_records

  def __getitem__(self, record_key: SupportsIndex) -> bytes:
    return self._reader[record_key]

  def __getstate__(self) -> dict[str, Any]:
    state = self.__dict__.copy()
    del state['_reader']
    return state

  def __setstate__(self, state) -> None:
    self.__dict__.update(state)
    self._reader = BagReader(self._path)

  def __repr__(self) -> str:
    return f'BagDataSource(path={self._path!r}'

def encode_message(element: str, fixed_fen: bool = True):
  fen, move = constants.CODERS['behavioral_cloning'].decode(element)
  if fixed_fen: 
    fen = convert_fen_to_fixed_length(fen)
  conversation = [
    {
      "content": f"Find the best UCI chess move for the following FEN position: {fen}",
      "role": "user"
    },
    {
      "content": f"{move}",
      "role": "assistant"
    },
  ]
  message = {
    'messages': conversation
  }
  return message

def generate_chunk(reader, start_idx, chunk_size):
    """
    Return a Dataset of (prompt, response) pairs from `reader` [start_idx : start_idx + chunk_size].
    """
    end_idx = min(start_idx + chunk_size, len(reader))
    prompts = []
    responses = []
    for i in range(start_idx, end_idx):
        fen, move = constants.CODERS['behavioral_cloning'].decode(reader[i])
        prompts.append(f"Find the best UCI chess move for the following FEN position: {fen}")
        responses.append(move)

    return Dataset.from_dict({
        'prompt': prompts,
        'response': responses,
    })

def convert_fen_to_fixed_length(fen: str) -> str:
    # Split the FEN string into components
    parts = fen.split()

    # Step 1: Convert board representation (64 characters)
    board = parts[0]
    fixed_board = ''
    for char in board:
        if char.isdigit():
            fixed_board += '.' * int(char)  # Replace numbers with dots
        elif char != '/':  # Ignore slashes separating ranks
            fixed_board += char

    # Ensure the board representation is exactly 64 characters
    if len(fixed_board) != 64:
        raise ValueError("Invalid FEN board representation")

    # Step 2: Convert active player (1 character)
    active_player = parts[1]  # 'w' or 'b'

    # Step 3: Convert castling availability (4 characters)
    castling = parts[2]
    fixed_castling = castling if castling != '-' else ''
    fixed_castling = fixed_castling.ljust(4, '.')  # Pad with '.' to make 4 characters

    # Step 4: Convert en passant target (2 characters)
    en_passant = parts[3]
    fixed_en_passant = en_passant if en_passant != '-' else '-.'

    # Step 5: Convert halfmove clock (2 characters)
    halfmove_clock = parts[4]
    fixed_halfmove_clock = halfmove_clock.rjust(2, '.')

    # Step 6: Convert fullmove number (3 characters)
    fullmove_number = parts[5]
    fixed_fullmove_number = fullmove_number.rjust(3, '.')

    # Combine all parts into the fixed-length FEN format
    fixed_fen = (
        fixed_board + ' ' +
        active_player + ' ' +
        fixed_castling + ' ' +
        fixed_en_passant + ' ' +
        fixed_halfmove_clock + ' ' +
        fixed_fullmove_number
    )

    return fixed_fen

if __name__ == '__main__':
  train_r = BagReader('../data/behavioral_cloning_data_train.bag')
  val_r = BagReader('../data/test/behavioral_cloning_data_test.bag')
  print("Number of positions", len(train_r))
  dataset_sizes = {
    'small_fixed': 500_000,
    'medium_fixed': 5_000_000,
    'large_fixed': 50_000_000,
    'all_fixed': -1,  # 500_000_000
  }
  chunk_size = 50_000
  repo_id = 'j316chuck/chess_rl'

  val_prompts = []
  val_responses = []
  for i in range(len(val_r)):
    fen, move = constants.CODERS['behavioral_cloning'].decode(val_r[i])
    val_prompts.append(f"You are an expert chess player. Find the best UCI chess move for the following FEN position: {fen}")
    val_responses.append(f"{move}")
  val_dataset = Dataset.from_dict({
    'prompt': val_prompts,
    'response': val_responses,
  })

  for dataset_name, max_records in dataset_sizes.items():
    print(f"Processing {dataset_name} dataset...")

    # If max_records == -1, that means "all" (the entire train).
    if max_records < 0:
      max_records = len(train_r)
    num_records = min(max_records, len(train_r))
    num_chunks = (num_records + chunk_size - 1) // chunk_size

    # We can push the validation dataset now if desired:
    val_dataset.push_to_hub(
        repo_id=repo_id,
        config_name=dataset_name,
        split='validation',
        private=True
    )

    cumulative_dataset = None
    for c in tqdm.trange(num_chunks):
      start_idx = c * chunk_size
      chunk_ds = generate_chunk(train_r, start_idx, chunk_size)

      if cumulative_dataset is None:
        cumulative_dataset = chunk_ds
      else:
        # We just append the new chunk to the old dataset
        cumulative_dataset = concatenate_datasets([cumulative_dataset, chunk_ds])
    # Now push the updated dataset to the hub
    cumulative_dataset.push_to_hub(
        repo_id=repo_id,
        config_name=dataset_name,
        split='train',
        private=True
    )

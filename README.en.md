# Pulat

[Türkçe](README.md) | **English**

A backgammon AI that learns purely by playing against itself. The network is given no strategy knowledge: it only knows the rules, looks at the raw board, and learns from millions of self-play games (TD(λ), the TD-Gammon method).

Scoring follows Turkish tavla rules: a win is 1 point, a gammon (mars) is 2 points, and there is no doubling cube.

## Features

- **Rules engine** (`engine.py`): generates every legal move, including the rules that you must play as many dice as possible and, if only one die can be played, the higher one.
- **Fast Rust engine** (`fastengine/`): the same rules, with batched move generation across all CPU cores (~2 million positions per second).
- **Neural network** (`model.py`): 196 inputs, two hidden layers of 256; outputs the probabilities of winning, winning a gammon and losing a gammon.
- **Training**: `train.py` (pure Python) and `train_fast.py` (Rust + GPU, thousands of games in parallel).
- **Lookahead** (`search.py`): a 2-ply search that considers all 21 possible opponent rolls.
- **Play**: in the terminal (`play.py`) or by clicking in the browser (`web.py`).
- **Evaluation**: tools to compare against a random player, older versions and GNU Backgammon.

Pretrained model: `checkpoints/pulat_best.pt`. It beats a random player 99.9% of the time and picks the expert-approved move for 13 of the 15 opening rolls (`opening_test.py`).

## Installation

```bash
pip install -r requirements.txt
cd fastengine && cargo build --release && cd ..
```

The Rust engine is required by `train_fast.py`, `web.py`, `search.py` and the evaluation tools. `engine.py`, `play.py` and `train.py` work without it. The GPU is used if CUDA is available, otherwise the CPU.

## Usage

```bash
python play.py                         # play against Pulat in the terminal
python web.py                          # play in the browser: http://localhost:8765
python train_fast.py --hours 3         # fast training from scratch (checkpoints_fast/)
python train.py --hours 2              # pure Python training (checkpoints/)
python opening_test.py checkpoints/pulat_best.pt
python compare_ply.py checkpoints/pulat_best.pt 1000
python test_fast.py                    # check the Rust and Python engines generate the same moves
```

The game interfaces and console output are in Turkish.

### Comparing with GNU Backgammon

`vs_gnubg.py` plays Pulat against GNU Backgammon and measures its move quality (PR). GNU Backgammon must be installed. The path to `gnubg-cli` can be set with the `GNUBG` environment variable:

```bash
python vs_gnubg.py --model checkpoints/pulat_best.pt --games 1000 --depth 2 --gnubg-plies 2
```

## Board format

A position is 28 integers, always from the point of view of the player to move:

| Index | Meaning |
|---|---|
| 0..23 | points: `+` my checkers, `−` opponent's checkers. I move from 23 toward 0; my home board is 0..5 |
| 24 | my checkers on the bar |
| 25 | opponent's checkers on the bar |
| 26 | my checkers borne off |
| 27 | opponent's checkers borne off |

The network evaluates the position after a move, from the point of view of the player who just moved. Expected points (equity): `2·win − 1 + gammon_win − gammon_loss`.

## License

MIT

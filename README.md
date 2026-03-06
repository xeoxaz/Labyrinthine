# Labyrinthine

Linux-first Rust terminal maze runner with Q-learning, auto-solving, and progressive wide-screen levels.

## Current implementation

- Seeded maze generation via recursive backtracker
- Playable pawn mode in terminal
- Turn-based movement (one keypress = one move)
- Manual, auto-solve, and **ML-Agent toggle modes** (`M` key)
- **Q-learning reinforcement agent** that learns to solve mazes
- **Progressive difficulty levels** with increasingly complex mazes
- Win condition when pawn reaches exit
- Timer that starts on first move and freezes on solve
- Host resource policy flags (`--max-mode`, `--threads`)

## Run

```bash
# Standard gameplay
cargo run -- play --width 39 --height 21 --seed 42 --max-mode

# Start with ML agent (toggle modes with M)
cargo run -- play --width 15 --height 9 --seed 7

# Require GPU-ready ML runtime and fail if no GPU backend is detected
cargo run -- play --require-gpu

# Force CPU-only ML runtime
cargo run -- play --cpu-only
```

## Controls

- `W/A/S/D` or arrow keys: move pawn
- `M`: cycle through modes (Manual → AutoSolve → ML-Agent → Manual)
- `N`: advance to next level (after solving current level)
- `Q` or `Esc`: quit

## Modes

- **Manual**: Player controls pawn directly
- **AutoSolve**: Shows BFS shortest path solution with `.` markers
- **ML-Agent**: Q-learning agent solves the maze (animated, learns from experience)

## Level Progression

- Each level increases maze complexity (wall density increases gradually)
- Maze dimensions grow: ~9×9 → 11×11 → 13×13 → ...
- ML agent difficulty multiplier increases per level
- Tracks player wins (P) and agent wins (A) at bottom of screen

## Generate-only output

```bash
cargo run -- generate --width 25 --height 15 --seed 7
```

## Resource tuning

- `--max-mode`: use near-full available CPU threads
- `--threads N`: explicit thread cap

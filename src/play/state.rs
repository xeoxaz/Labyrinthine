use std::time::Instant;

use crate::core::grid::Maze;

const TRAIL_LIMIT: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlMode {
    Manual,
    AutoSolve,
    MLAgent,
}

#[derive(Clone, Debug)]
pub struct PlayerState {
    pub position: (usize, usize),
    pub control_mode: ControlMode,
    pub won: bool,
    pub steps: u64,
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub maze: Maze,
    pub player: PlayerState,
    pub created_at: Option<Instant>,
    pub autosolve_path: Vec<(usize, usize)>,
    pub autosolve_index: usize,
    pub final_elapsed_secs: Option<u64>,
    pub trail: Vec<(usize, usize)>,
}

impl GameState {
    pub fn new(maze: Maze, autosolve_path: Vec<(usize, usize)>) -> Self {
        let start = maze.start;
        Self {
            maze,
            player: PlayerState {
                position: start,
                control_mode: ControlMode::Manual,
                won: false,
                steps: 0,
            },
            created_at: None,
            autosolve_path,
            autosolve_index: 0,
            final_elapsed_secs: None,
            trail: Vec::with_capacity(TRAIL_LIMIT),
        }
    }

    pub fn elapsed_secs(&self) -> u64 {
        if let Some(final_time) = self.final_elapsed_secs {
            final_time
        } else if let Some(started_at) = self.created_at {
            started_at.elapsed().as_secs()
        } else {
            0
        }
    }

    pub fn mark_won(&mut self) {
        self.player.won = true;
        if let Some(started_at) = self.created_at {
            self.final_elapsed_secs = Some(started_at.elapsed().as_secs());
        }
    }

    pub fn start_timer_if_needed(&mut self) {
        if self.created_at.is_none() {
            self.created_at = Some(Instant::now());
        }
    }

    pub fn move_player_to(&mut self, next: (usize, usize)) {
        if next == self.player.position {
            return;
        }

        self.trail.push(self.player.position);
        if self.trail.len() > TRAIL_LIMIT {
            self.trail.remove(0);
        }

        self.player.position = next;
        self.player.steps += 1;
    }

    pub fn progress_ratio(&self) -> f32 {
        let total = manhattan(self.maze.start, self.maze.exit);
        if total == 0 {
            return 1.0;
        }

        let remaining = manhattan(self.player.position, self.maze.exit);
        ((total.saturating_sub(remaining)) as f32 / total as f32).clamp(0.0, 1.0)
    }
}

fn manhattan(a: (usize, usize), b: (usize, usize)) -> usize {
    a.0.abs_diff(b.0) + a.1.abs_diff(b.1)
}

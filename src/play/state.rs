use std::time::Instant;

use crate::core::grid::Maze;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlMode {
    Manual,
    AutoSolve,
    MLAgent,
}

#[derive(Debug)]
pub struct PlayerState {
    pub position: (usize, usize),
    pub control_mode: ControlMode,
    pub won: bool,
    pub steps: u64,
}

#[derive(Debug)]
pub struct GameState {
    pub maze: Maze,
    pub player: PlayerState,
    pub created_at: Option<Instant>,
    pub autosolve_path: Vec<(usize, usize)>,
    pub autosolve_index: usize,
    pub final_elapsed_secs: Option<u64>,
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
}

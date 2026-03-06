use crate::core::grid::Maze;
use crate::core::generator::generate_recursive_backtracker;
use crate::ml::{QLearningAgent, LevelManager};
use crate::play::state::ControlMode;
use crate::play::state::GameState;
use crate::play::pathing;
use crate::play::ml_solver::MLSolver;

/// Level-based gameplay state handling progression
pub struct LevelGame {
    pub level_manager: LevelManager,
    pub current_game: GameState,
    pub ml_solver: Option<MLSolver>,
    pub maze_seed: u64,
    pub status_message: Option<String>,
    pub max_width: usize,
    pub max_height: usize,
}

impl LevelGame {
    pub fn new(initial_seed: u64, max_width: usize, max_height: usize) -> Self {
        let mut level_game = Self {
            level_manager: LevelManager::new(),
            current_game: GameState::new(Maze::new(2, 2), Vec::new()),
            ml_solver: None,
            maze_seed: initial_seed,
            status_message: None,
            max_width: max_width.max(2),
            max_height: max_height.max(2),
        };
        let mut game = level_game.build_game_for_level(1);
        game.player.control_mode = ControlMode::MLAgent;
        level_game.current_game = game;
        level_game.create_ml_solver();
        level_game
    }

    /// Generate a maze for a specific level with increased complexity
    fn generate_level_maze(level: usize, base_seed: u64, max_width: usize, max_height: usize) -> Maze {
        let max_width = max_width.max(2);
        let max_height = max_height.max(2);
        let (width, height) = Self::wide_dimensions(level, base_seed, max_width, max_height)
            .unwrap_or_else(|| Self::square_dimensions(level, max_width, max_height));
        let seed = Self::mix_seed(base_seed, level as u64 + 97);

        // Generate with adjusted parameters for higher difficulty
        generate_recursive_backtracker(width, height, seed)
    }

    /// Advance to next level, resetting game state but keeping learned model
    pub fn next_level(&mut self) {
        let next_mode = self.current_game.player.control_mode;
        self.level_manager.next_level();
        let level = self.level_manager.current_level();
        self.current_game = self.build_game_for_level(level);
        self.current_game.player.control_mode = next_mode;
        self.status_message = None;

        // Update agent difficulty if ML solver exists
        if let Some(solver) = &mut self.ml_solver {
            solver.set_difficulty(self.level_manager.get_agent_difficulty());
        }
    }

    /// Create or switch to ML solver
    pub fn create_ml_solver(&mut self) {
        let mut agent = QLearningAgent::new();
        agent.set_difficulty(self.level_manager.get_agent_difficulty());
        self.ml_solver = Some(MLSolver::new(agent));
    }

    pub fn get_complexity(&self) -> f32 {
        self.level_manager.get_complexity()
    }

    pub fn ml_time_limit_secs(&self) -> u64 {
        self.level_manager.ml_time_limit_secs()
    }

    pub fn ml_step_limit(&self) -> u64 {
        self.level_manager.ml_step_limit(&self.current_game.maze)
    }

    pub fn restart_ml_attempt(&mut self, reason: impl Into<String>) {
        let level = self.level_manager.current_level();
        let mut game = self.build_game_for_level(level);
        game.player.control_mode = ControlMode::MLAgent;
        self.current_game = game;
        self.status_message = Some(reason.into());
    }

    fn build_game_for_level(&self, level: usize) -> GameState {
        let maze = Self::generate_level_maze(level, self.maze_seed, self.max_width, self.max_height);
        let autosolve_path = pathing::build_autosolve_path(&maze);
        GameState::new(maze, autosolve_path)
    }

    fn mix_seed(base_seed: u64, salt: u64) -> u64 {
        base_seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(salt.wrapping_mul(0xBF58_476D_1CE4_E5B9))
            .rotate_left((salt % 31) as u32)
    }

    fn square_dimensions(level: usize, max_width: usize, max_height: usize) -> (usize, usize) {
        let max_size = max_width.min(max_height).max(2);
        let target_size = (6 + level * 2).min(max_size).max(2);
        let min_size = target_size.saturating_sub(2).max(2);
        let size = target_size.max(min_size);
        (size, size)
    }

    fn wide_dimensions(level: usize, base_seed: u64, max_width: usize, max_height: usize) -> Option<(usize, usize)> {
        let max_wide_width = max_width.min(max_height.saturating_mul(16) / 9).max(2);
        if max_wide_width < 4 {
            return None;
        }

        let target_width = (12 + level * 4).min(max_wide_width).max(4);
        let min_width = target_width.saturating_sub(4).max(4);
        let width_range = target_width - min_width + 1;
        let mut width = min_width
            + (Self::mix_seed(base_seed, level as u64 * 13 + 7) as usize % width_range);
        width = width.min(max_wide_width).max(4);

        let mut height = width.saturating_mul(9) / 16;
        height = height.max(2).min(max_height);

        if height > width {
            return None;
        }

        let corrected_width = (height.saturating_mul(16) / 9).max(width.saturating_sub(1));
        let width = corrected_width.min(max_wide_width).max(height);

        if width < height {
            None
        } else {
            Some((width, height))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_in_ml_agent_mode() {
        let level_game = LevelGame::new(123, 20, 12);

        assert_eq!(level_game.current_game.player.control_mode, ControlMode::MLAgent);
        assert!(level_game.ml_solver.is_some());
    }

    #[test]
    fn maze_dimensions_stay_within_bounds() {
        let mut level_game = LevelGame::new(321, 10, 8);

        assert!(level_game.current_game.maze.width() <= 10);
        assert!(level_game.current_game.maze.height() <= 8);

        level_game.next_level();

        assert!(level_game.current_game.maze.width() <= 10);
        assert!(level_game.current_game.maze.height() <= 8);
    }

    #[test]
    fn maze_is_never_taller_than_wide() {
        let mut level_game = LevelGame::new(999, 20, 12);

        for _ in 0..5 {
            assert!(level_game.current_game.maze.width() >= level_game.current_game.maze.height());
            level_game.next_level();
        }
    }

    #[test]
    fn maze_uses_square_or_wide_layouts() {
        let mut level_game = LevelGame::new(777, 30, 14);

        for _ in 0..5 {
            let width = level_game.current_game.maze.width();
            let height = level_game.current_game.maze.height();
            let is_square = width == height;
            let is_wide = width > height;

            assert!(is_square || is_wide);
            level_game.next_level();
        }
    }

    #[test]
    fn wide_layout_is_preferred_on_wide_terminal() {
        let level_game = LevelGame::new(888, 40, 14);

        assert!(level_game.current_game.maze.width() > level_game.current_game.maze.height());
    }
}

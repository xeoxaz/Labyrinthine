use crate::core::grid::Maze;
use crate::core::generator::generate_recursive_backtracker;
use crate::ml::{QLearningAgent, LevelManager};
use crate::play::ml_solver::MlHudStats;
use crate::play::state::ControlMode;
use crate::play::state::GameState;
use crate::play::pathing;
use crate::play::ml_solver::MLSolver;

#[derive(Clone, Debug)]
pub struct LevelLoadingState {
    pub level: usize,
    pub template_game: GameState,
    pub training_game: GameState,
    pub warmup_episodes_done: u32,
    pub warmup_episodes_total: u32,
    pub last_summary: String,
}

impl LevelLoadingState {
    pub fn progress_ratio(&self) -> f32 {
        if self.warmup_episodes_total == 0 {
            return 1.0;
        }

        let in_run = self.training_game.progress_ratio();
        ((self.warmup_episodes_done as f32 + in_run) / self.warmup_episodes_total as f32)
            .clamp(0.0, 1.0)
    }
}

/// Level-based gameplay state handling progression
pub struct LevelGame {
    pub level_manager: LevelManager,
    pub current_game: GameState,
    pub ml_solver: Option<MLSolver>,
    pub loading_level: Option<LevelLoadingState>,
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
            loading_level: None,
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

    pub fn begin_initial_loading(&mut self) {
        let level = self.level_manager.current_level();
        let mut template_game = self.current_game.clone();
        template_game.player.control_mode = ControlMode::MLAgent;
        let training_game = template_game.clone();
        let total_warmups = Self::warmup_episodes_for(level);

        if let Some(solver) = &mut self.ml_solver {
            solver.set_difficulty(self.level_manager.get_agent_difficulty());
            solver.begin_new_episode();
        }

        self.loading_level = Some(LevelLoadingState {
            level,
            template_game,
            training_game,
            warmup_episodes_done: 0,
            warmup_episodes_total: total_warmups,
            last_summary: "Checking the first maze and warming up the agent".to_string(),
        });
        self.status_message = Some(format!("Loading level {}", level));
    }

    /// Generate a maze for a specific level with increased complexity
    fn generate_level_maze(level: usize, base_seed: u64, max_width: usize, max_height: usize) -> Maze {
        let max_width = max_width.max(2);
        let max_height = max_height.max(2);
        let (width, height) = Self::wide_dimensions(level, base_seed, max_width, max_height)
            .unwrap_or_else(|| Self::best_effort_wide_dimensions(max_width, max_height));
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
        self.loading_level = None;
        self.status_message = None;

        // Update agent difficulty if ML solver exists
        if let Some(solver) = &mut self.ml_solver {
            solver.set_difficulty(self.level_manager.get_agent_difficulty());
            if next_mode == ControlMode::MLAgent {
                solver.begin_new_episode();
            }
        }
    }

    /// Create or switch to ML solver
    pub fn create_ml_solver(&mut self) {
        let mut agent = QLearningAgent::new();
        agent.set_difficulty(self.level_manager.get_agent_difficulty());
        self.ml_solver = Some(MLSolver::new(agent));
    }

    pub fn ml_time_limit_secs(&self) -> u64 {
        self.level_manager
            .ml_time_limit_secs(&self.current_game.maze, self.current_game.autosolve_path.len())
    }

    pub fn ml_step_limit(&self) -> u64 {
        self.level_manager
            .ml_step_limit(&self.current_game.maze, self.current_game.autosolve_path.len())
    }

    pub fn ml_hud_stats(&self) -> Option<MlHudStats> {
        self.ml_solver.as_ref().map(MLSolver::hud_stats)
    }

    pub fn begin_next_level_loading(&mut self) {
        self.level_manager.next_level();
        let level = self.level_manager.current_level();
        let mut template_game = self.build_game_for_level(level);
        template_game.player.control_mode = ControlMode::MLAgent;
        let training_game = template_game.clone();
        let total_warmups = Self::warmup_episodes_for(level);

        if let Some(solver) = &mut self.ml_solver {
            solver.set_difficulty(self.level_manager.get_agent_difficulty());
            solver.begin_new_episode();
        }

        self.loading_level = Some(LevelLoadingState {
            level,
            template_game,
            training_game,
            warmup_episodes_done: 0,
            warmup_episodes_total: total_warmups,
            last_summary: "Building the next maze and warming up the agent".to_string(),
        });
        self.status_message = Some(format!("Loading level {}", level));
    }

    pub fn finish_loading_level(&mut self, loading: LevelLoadingState) {
        self.current_game = loading.template_game;
        self.loading_level = None;
        self.status_message = Some(format!("Level {} ready", loading.level));

        if let Some(solver) = &mut self.ml_solver {
            solver.begin_new_episode();
        }
    }

    fn warmup_episodes_for(level: usize) -> u32 {
        (3 + (level / 2) as u32).min(8)
    }

    pub fn restart_ml_attempt(&mut self, reason: impl Into<String>) {
        let level = self.level_manager.current_level();
        let mut game = self.build_game_for_level(level);
        game.player.control_mode = ControlMode::MLAgent;
        self.current_game = game;
        self.status_message = Some(reason.into());
        if let Some(solver) = &mut self.ml_solver {
            solver.begin_new_episode();
        }
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

    fn best_effort_wide_dimensions(max_width: usize, max_height: usize) -> (usize, usize) {
        let width = max_width.max(2);
        let height = ((width.saturating_mul(9)) / 16).max(2).min(max_height.max(2));

        if width >= height {
            (width, height)
        } else {
            (height, height.max(2))
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
            assert!(width > height);
            level_game.next_level();
        }
    }

    #[test]
    fn wide_layout_is_preferred_on_wide_terminal() {
        let level_game = LevelGame::new(888, 40, 14);

        assert!(level_game.current_game.maze.width() > level_game.current_game.maze.height());
    }

    #[test]
    fn wide_layout_tracks_sixteen_by_nine_ratio() {
        let mut level_game = LevelGame::new(555, 40, 16);

        for _ in 0..5 {
            let width = level_game.current_game.maze.width() as i64;
            let height = level_game.current_game.maze.height() as i64;
            let ratio_error = (width * 9 - height * 16).abs();

            assert!(ratio_error <= 16);
            level_game.next_level();
        }
    }

    #[test]
    fn loading_state_stages_next_level_for_ml() {
        let mut level_game = LevelGame::new(123, 24, 12);
        let current_level = level_game.level_manager.current_level();

        level_game.begin_next_level_loading();

        let loading = level_game.loading_level.as_ref().expect("loading state");
        assert_eq!(loading.level, current_level + 1);
        assert_eq!(level_game.level_manager.current_level(), current_level + 1);
        assert!(loading.warmup_episodes_total >= 3);
        assert_eq!(loading.template_game.player.control_mode, ControlMode::MLAgent);
    }

    #[test]
    fn initial_loading_stages_first_level_without_advancing() {
        let mut level_game = LevelGame::new(123, 24, 12);

        level_game.begin_initial_loading();

        let loading = level_game.loading_level.as_ref().expect("loading state");
        assert_eq!(loading.level, 1);
        assert_eq!(level_game.level_manager.current_level(), 1);
        assert!(loading.warmup_episodes_total >= 3);
        assert_eq!(loading.template_game.player.control_mode, ControlMode::MLAgent);
    }
}

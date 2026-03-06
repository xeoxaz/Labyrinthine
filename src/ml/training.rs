use crate::core::grid::Maze;

pub struct LevelManager {
    current_level: usize,
    player_wins: usize,
    agent_wins: usize,
}

impl LevelManager {
    pub fn new() -> Self {
        Self {
            current_level: 1,
            player_wins: 0,
            agent_wins: 0,
        }
    }

    pub fn current_level(&self) -> usize {
        self.current_level
    }

    pub fn player_wins(&self) -> usize {
        self.player_wins
    }

    pub fn agent_wins(&self) -> usize {
        self.agent_wins
    }

    /// Record a player win and check for level progression
    pub fn record_player_win(&mut self) {
        self.player_wins += 1;
    }

    /// Record an agent win and check for level progression
    pub fn record_agent_win(&mut self) {
        self.agent_wins += 1;
    }

    /// Advance to next level with increased difficulty
    pub fn next_level(&mut self) {
        self.current_level += 1;
        self.player_wins = 0;
        self.agent_wins = 0;
    }

    /// Get agent difficulty multiplier (affects reward scaling)
    pub fn get_agent_difficulty(&self) -> f32 {
        1.0 + (self.current_level as f32 - 1.0) * 0.15
    }

    pub fn ml_time_limit_secs(&self, maze: &Maze, shortest_path_len: usize) -> u64 {
        let area = (maze.width() * maze.height()) as u64;
        let path = shortest_path_len.max(2) as u64;
        let area_budget = area / 18;
        let path_budget = path / 3;

        12 + self.current_level as u64 * 3 + area_budget + path_budget
    }

    pub fn ml_step_limit(&self, maze: &Maze, shortest_path_len: usize) -> u64 {
        let area = (maze.width() * maze.height()) as u64;
        let path = shortest_path_len.max(2) as u64;
        let area_budget = area * 3;
        let path_budget = path * 8;

        80 + self.current_level as u64 * 60 + area_budget + path_budget
    }
}

impl Default for LevelManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_progression() {
        let mut manager = LevelManager::new();
        assert_eq!(manager.current_level(), 1);
        assert_eq!(manager.get_agent_difficulty(), 1.0);

        manager.next_level();
        assert_eq!(manager.current_level(), 2);
        assert!(manager.get_agent_difficulty() > 1.0);
    }

    #[test]
    fn difficulty_scaling() {
        let manager = LevelManager::new();
        let diff1 = manager.get_agent_difficulty();

        let mut manager2 = LevelManager::new();
        manager2.next_level();
        manager2.next_level();
        let diff3 = manager2.get_agent_difficulty();

        assert!(diff3 > diff1);
    }

    #[test]
    fn ml_limits_scale_with_level() {
        let maze = Maze::new(11, 11);
        let level1 = LevelManager::new();
        let mut level3 = LevelManager::new();
        level3.next_level();
        level3.next_level();
        let path_len = 32;

        assert!(level3.ml_time_limit_secs(&maze, path_len) > level1.ml_time_limit_secs(&maze, path_len));
        assert!(level3.ml_step_limit(&maze, path_len) > level1.ml_step_limit(&maze, path_len));
    }

    #[test]
    fn longer_paths_get_more_budget() {
        let maze = Maze::new(20, 10);
        let manager = LevelManager::new();

        assert!(manager.ml_time_limit_secs(&maze, 80) > manager.ml_time_limit_secs(&maze, 20));
        assert!(manager.ml_step_limit(&maze, 80) > manager.ml_step_limit(&maze, 20));
    }
}

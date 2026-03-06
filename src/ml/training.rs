use crate::core::grid::Maze;
use crate::ml::agent::QState;

pub struct LevelManager {
    current_level: usize,
    player_wins: usize,
    agent_wins: usize,
    base_complexity: f32,
}

impl LevelManager {
    pub fn new() -> Self {
        Self {
            current_level: 1,
            player_wins: 0,
            agent_wins: 0,
            base_complexity: 0.5,
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

    /// Get complexity multiplier based on current level
    /// Complexity affects wall density in maze generation
    pub fn get_complexity(&self) -> f32 {
        self.base_complexity + (self.current_level as f32 - 1.0) * 0.1
    }

    /// Get agent difficulty multiplier (affects reward scaling)
    pub fn get_agent_difficulty(&self) -> f32 {
        1.0 + (self.current_level as f32 - 1.0) * 0.15
    }

    pub fn ml_time_limit_secs(&self) -> u64 {
        6 + self.current_level as u64 * 2
    }

    pub fn ml_step_limit(&self, maze: &Maze) -> u64 {
        let area = (maze.width() * maze.height()) as u64;
        area + self.current_level as u64 * 40
    }
}

impl Default for LevelManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert game position to Q-learning state
pub fn game_state_to_q_state(position: (usize, usize), direction_idx: usize) -> QState {
    QState(position.0, position.1, direction_idx)
}

/// Compute manhattan distance heuristic
pub fn manhattan_distance(a: (usize, usize), b: (usize, usize)) -> f32 {
    (a.0.abs_diff(b.0) + a.1.abs_diff(b.1)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_progression() {
        let mut manager = LevelManager::new();
        assert_eq!(manager.current_level(), 1);
        assert!(manager.get_complexity() < 0.6);

        manager.next_level();
        assert_eq!(manager.current_level(), 2);
        assert!(manager.get_complexity() > 0.5);
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

        assert!(level3.ml_time_limit_secs() > level1.ml_time_limit_secs());
        assert!(level3.ml_step_limit(&maze) > level1.ml_step_limit(&maze));
    }
}

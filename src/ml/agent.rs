use std::collections::HashMap;

use rand::Rng;

use crate::core::grid::Direction;

/// Q-Learning state: discretized as (x, y, direction_facing)
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct QState(pub usize, pub usize, pub usize);

/// Actions: 0=North, 1=East, 2=South, 3=West
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentAction {
    North,
    East,
    South,
    West,
}

impl AgentAction {
    pub fn to_direction(self) -> Direction {
        match self {
            AgentAction::North => Direction::North,
            AgentAction::East => Direction::East,
            AgentAction::South => Direction::South,
            AgentAction::West => Direction::West,
        }
    }

    pub fn index(self) -> usize {
        match self {
            AgentAction::North => 0,
            AgentAction::East => 1,
            AgentAction::South => 2,
            AgentAction::West => 3,
        }
    }
}

pub struct QLearningAgent {
    /// Q-table: maps (state, action) -> estimated reward
    q_table: HashMap<(QState, usize), f32>,

    /// Learning rate (alpha)
    learning_rate: f32,

    /// Discount factor (gamma)
    discount: f32,

    /// Exploration rate (epsilon) - starts high, decays over training
    epsilon: f32,

    /// Epsilon decay per episode
    epsilon_decay: f32,

    /// Minimum epsilon (explore at least this much)
    epsilon_min: f32,

    /// Difficulty/penalty multiplier
    difficulty: f32,
}

impl QLearningAgent {
    /// Create a new Q-learning agent with default parameters
    pub fn new() -> Self {
        Self {
            q_table: HashMap::new(),
            learning_rate: 0.1,
            discount: 0.99,
            epsilon: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.05,
            difficulty: 1.0,
        }
    }

    /// Set difficulty multiplier (1.0 = normal, 1.5 = harder)
    pub fn set_difficulty(&mut self, difficulty: f32) {
        self.difficulty = difficulty.max(0.5).min(5.0);
    }

    /// Get the next action using epsilon-greedy strategy
    pub fn select_action(
        &self,
        state: QState,
        valid_actions: &[AgentAction],
        rng: &mut impl Rng,
    ) -> AgentAction {
        let actions = if valid_actions.is_empty() {
            &AgentAction::ALL
        } else {
            valid_actions
        };

        if rng.random::<f32>() < self.epsilon {
            // Explore: random action
            let idx = rng.random_range(0..actions.len());
            actions[idx]
        } else {
            // Exploit: best known action
            self.best_action(state, actions)
        }
    }

    /// Get the best known action for a state
    fn best_action(&self, state: QState, valid_actions: &[AgentAction]) -> AgentAction {
        let mut best_action = valid_actions.first().copied().unwrap_or(AgentAction::North);
        let mut best_value = f32::NEG_INFINITY;

        for action in valid_actions {
            let q_val = self
                .q_table
                .get(&(state, action.index()))
                .copied()
                .unwrap_or(0.0);
            if q_val > best_value {
                best_value = q_val;
                best_action = *action;
            }
        }

        best_action
    }

    /// Get Q-value for (state, action)
    pub fn get_q_value(&self, state: QState, action: AgentAction) -> f32 {
        self.q_table
            .get(&(state, action.index()))
            .copied()
            .unwrap_or(0.0)
    }

    /// Update Q-value using Bellman equation
    /// reward: immediate reward received
    /// next_state: the state we transitioned to
    /// terminal: whether the episode ended
    pub fn update_q_value(
        &mut self,
        state: QState,
        action: AgentAction,
        reward: f32,
        next_state: QState,
        terminal: bool,
    ) {
        let action_idx = action.index();
        let current_q = self.get_q_value(state, action);

        let max_next_q = if terminal {
            0.0
        } else {
            (0..4)
                .map(|a_idx| self.q_table.get(&(next_state, a_idx)).copied().unwrap_or(0.0))
                .fold(f32::NEG_INFINITY, f32::max)
        };

        let new_q = current_q
            + self.learning_rate * (reward + self.discount * max_next_q - current_q);

        self.q_table.insert((state, action_idx), new_q);
    }

    /// Compute reward based on movement and progress
    /// positive: getting closer to exit, reaching it
    /// negative: hitting walls, idle moves with difficulty penalty
    pub fn compute_reward(
        &self,
        moved: bool,
        backtracked: bool,
        new_dist_to_exit: usize,
        old_dist_to_exit: usize,
        reached_exit: bool,
    ) -> f32 {
        if reached_exit {
            // Major reward for solving the maze
            100.0 * self.difficulty
        } else if backtracked {
            // Stronger penalty for undoing the previous move.
            -2.0 * self.difficulty
        } else if !moved {
            // Penalty for hitting wall (harder penalties at higher difficulty)
            -1.0 * self.difficulty
        } else if new_dist_to_exit < old_dist_to_exit {
            // Reward true maze progress rather than straight-line movement.
            let gain = old_dist_to_exit.saturating_sub(new_dist_to_exit) as f32;
            (0.4 + gain * 0.2) * self.difficulty
        } else if new_dist_to_exit > old_dist_to_exit {
            let loss = new_dist_to_exit.saturating_sub(old_dist_to_exit) as f32;
            (-0.3 - loss * 0.15) * self.difficulty
        } else {
            -0.2 * self.difficulty
        }
    }

    pub fn failure_penalty(&self) -> f32 {
        -25.0 * self.difficulty
    }

    /// Decay epsilon after each episode
    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }
}

impl Default for QLearningAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentAction {
    pub const ALL: [AgentAction; 4] = [
        AgentAction::North,
        AgentAction::East,
        AgentAction::South,
        AgentAction::West,
    ];
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn agent_action_conversions() {
        assert_eq!(AgentAction::North.to_direction(), Direction::North);
        assert_eq!(AgentAction::East.to_direction(), Direction::East);
        assert_eq!(AgentAction::South.to_direction(), Direction::South);
    }

    #[test]
    fn q_learning_updates() {
        let mut agent = QLearningAgent::new();
        let state = QState(0, 0, 0);
        let next_state = QState(1, 0, 0);

        // Update Q-value
        agent.update_q_value(state, AgentAction::East, 0.5, next_state, false);
        let q_val = agent.get_q_value(state, AgentAction::East);
        assert!(q_val > 0.0);

        // Reward for reaching exit
        let exit_reward = agent.compute_reward(true, false, 0, 1, true);
        assert_eq!(exit_reward, 100.0);
    }

    #[test]
    fn backtracking_penalty_is_stronger_than_idle_penalty() {
        let agent = QLearningAgent::new();

        let idle_penalty = agent.compute_reward(false, false, 3, 3, false);
        let backtrack_penalty = agent.compute_reward(true, true, 3, 2, false);

        assert!(backtrack_penalty < idle_penalty);
    }

    #[test]
    fn valid_action_mask_ignores_invalid_best_q() {
        let mut agent = QLearningAgent::new();
        let state = QState(0, 0, 0);
        agent.update_q_value(state, AgentAction::North, 10.0, state, false);
        agent.update_q_value(state, AgentAction::East, 1.0, state, false);

        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let action = agent.select_action(state, &[AgentAction::East], &mut rng);

        assert_eq!(action, AgentAction::East);
    }

    #[test]
    fn epsilon_decay() {
        let mut agent = QLearningAgent::new();
        let initial_eps = agent.epsilon();
        agent.decay_epsilon();
        assert!(agent.epsilon() < initial_eps);
    }

    #[test]
    fn failure_penalty_scales_with_difficulty() {
        let mut agent = QLearningAgent::new();
        let baseline = agent.failure_penalty();
        agent.set_difficulty(2.0);

        assert!(agent.failure_penalty() < baseline);
    }
}

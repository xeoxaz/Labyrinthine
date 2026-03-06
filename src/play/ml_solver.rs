use rand::Rng;
use crate::core::solver::shortest_path_distance;
use crate::ml::{QLearningAgent, AgentAction, QState};
use crate::play::state::GameState;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MlHudStats {
    pub episodes: u64,
    pub wins: u64,
    pub failures: u64,
    pub epsilon: f32,
    pub last_reward: f32,
    pub avg_reward: f32,
    pub episode_steps: u64,
    pub best_episode_steps: Option<u64>,
}

/// Handles ML agent decision making and movement
pub struct MLSolver {
    agent: QLearningAgent,
    last_state: Option<QState>,
    last_action: Option<AgentAction>,
    last_distance: usize,
    previous_position: Option<(usize, usize)>,
    episodes: u64,
    wins: u64,
    failures: u64,
    last_reward: f32,
    avg_reward: f32,
    episode_steps: u64,
    best_episode_steps: Option<u64>,
}

impl MLSolver {
    pub fn new(agent: QLearningAgent) -> Self {
        Self {
            agent,
            last_state: None,
            last_action: None,
            last_distance: 0,
            previous_position: None,
            episodes: 1,
            wins: 0,
            failures: 0,
            last_reward: 0.0,
            avg_reward: 0.0,
            episode_steps: 0,
            best_episode_steps: None,
        }
    }

    /// Get the next action for the ML agent to take
    pub fn step(&mut self, game: &GameState, rng: &mut impl Rng) -> AgentAction {
        let current_pos = game.player.position;
        let state = QState(current_pos.0, current_pos.1, 0);
        let valid_actions = valid_actions(game);

        // Select action using epsilon-greedy
        let action = self.agent.select_action(state, &valid_actions, rng);

        // Store for Q-learning update
        self.previous_position = self.last_state.map(|last| (last.0, last.1));
        self.last_state = Some(state);
        self.last_action = Some(action);
        self.last_distance = shortest_path_distance(&game.maze, current_pos, game.maze.exit)
            .unwrap_or(usize::MAX / 4);

        action
    }

    /// Update Q-learning based on the action outcome
    pub fn update_q_learning(&mut self, game: &GameState, moved: bool) {
        if let (Some(state), Some(action)) = (self.last_state, self.last_action) {
            let new_pos = game.player.position;
            let new_state = QState(new_pos.0, new_pos.1, 0);
            let new_distance = shortest_path_distance(&game.maze, new_pos, game.maze.exit)
                .unwrap_or(usize::MAX / 4);
            let reached_exit = game.player.won;
            let backtracked = moved && self.previous_position == Some(new_pos);

            let reward = self.agent.compute_reward(
                moved,
                backtracked,
                new_distance,
                self.last_distance,
                reached_exit,
            );

            self.record_reward_sample(reward);

            self.agent.update_q_value(state, action, reward, new_state, reached_exit);
        }
    }

    pub fn decay_epsilon(&mut self) {
        self.agent.decay_epsilon();
    }

    pub fn set_difficulty(&mut self, difficulty: f32) {
        self.agent.set_difficulty(difficulty);
    }

    pub fn hud_stats(&self) -> MlHudStats {
        MlHudStats {
            episodes: self.episodes,
            wins: self.wins,
            failures: self.failures,
            epsilon: self.agent.epsilon(),
            last_reward: self.last_reward,
            avg_reward: self.avg_reward,
            episode_steps: self.episode_steps,
            best_episode_steps: self.best_episode_steps,
        }
    }

    pub fn record_win(&mut self, steps: u64) {
        self.wins += 1;
        self.best_episode_steps = Some(
            self.best_episode_steps
                .map(|best| best.min(steps))
                .unwrap_or(steps),
        );
    }

    pub fn record_failure(&mut self, game: &GameState) {
        if let (Some(state), Some(action)) = (self.last_state, self.last_action) {
            let position = game.player.position;
            let next_state = QState(position.0, position.1, 0);
            self.agent.update_q_value(
                state,
                action,
                self.agent.failure_penalty(),
                next_state,
                true,
            );
        }

        self.failures += 1;
    }

    pub fn begin_new_episode(&mut self) {
        self.episodes += 1;
        self.last_state = None;
        self.last_action = None;
        self.last_distance = 0;
        self.previous_position = None;
        self.episode_steps = 0;
    }

    fn record_reward_sample(&mut self, reward: f32) {
        self.last_reward = reward;
        self.avg_reward = self.avg_reward * 0.92 + reward * 0.08;
        self.episode_steps += 1;
    }
}

fn valid_actions(game: &GameState) -> Vec<AgentAction> {
    AgentAction::ALL
        .into_iter()
        .filter(|action| game.maze.can_move(game.player.position, action.to_direction()).is_some())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::grid::{Direction, Maze};
    use crate::play::state::GameState;
    use rand::SeedableRng;

    #[test]
    fn ml_solver_makes_decisions() {
        let mut maze = Maze::new(2, 1);
        maze.remove_wall_between((0, 0), (1, 0), Direction::East);
        let game = GameState::new(maze, vec![]);

        let mut solver = MLSolver::new(QLearningAgent::new());
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let action = solver.step(&game, &mut rng);
        assert!((0..4).contains(&(action.index())));
    }

    #[test]
    fn record_failure_keeps_solver_operational() {
        let mut maze = Maze::new(2, 1);
        maze.remove_wall_between((0, 0), (1, 0), Direction::East);
        let game = GameState::new(maze, vec![]);

        let mut solver = MLSolver::new(QLearningAgent::new());
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);

        let _ = solver.step(&game, &mut rng);
        solver.record_failure(&game);

        let action = solver.step(&game, &mut rng);
        assert!((0..4).contains(&(action.index())));
    }

    #[test]
    fn step_selects_valid_action_only() {
        let mut maze = Maze::new(2, 2);
        maze.remove_wall_between((0, 0), (1, 0), Direction::East);
        let game = GameState::new(maze, vec![]);

        let mut solver = MLSolver::new(QLearningAgent::new());
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let action = solver.step(&game, &mut rng);

        assert_eq!(action, AgentAction::East);
    }

    #[test]
    fn telemetry_tracks_episode_transitions() {
        let mut solver = MLSolver::new(QLearningAgent::new());

        assert_eq!(solver.hud_stats().episodes, 1);
        solver.record_win(14);
        assert_eq!(solver.hud_stats().best_episode_steps, Some(14));

        solver.begin_new_episode();
        let stats = solver.hud_stats();
        assert_eq!(stats.episodes, 2);
        assert_eq!(stats.episode_steps, 0);
        assert_eq!(stats.wins, 1);
    }
}

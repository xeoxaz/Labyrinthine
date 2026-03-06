use rand::Rng;
use crate::core::grid::Direction;
use crate::ml::{QLearningAgent, AgentAction, QState};
use crate::ml::training::manhattan_distance;
use crate::play::state::GameState;

/// Handles ML agent decision making and movement
pub struct MLSolver {
    agent: QLearningAgent,
    last_state: Option<QState>,
    last_action: Option<AgentAction>,
    last_distance: f32,
    previous_position: Option<(usize, usize)>,
}

impl MLSolver {
    pub fn new(agent: QLearningAgent) -> Self {
        Self {
            agent,
            last_state: None,
            last_action: None,
            last_distance: 0.0,
            previous_position: None,
        }
    }

    /// Get the next action for the ML agent to take
    pub fn step(&mut self, game: &GameState, rng: &mut impl Rng) -> AgentAction {
        let current_pos = game.player.position;
        let state = QState(current_pos.0, current_pos.1, 0);

        // Select action using epsilon-greedy
        let action = self.agent.select_action(state, rng);

        // Store for Q-learning update
        self.previous_position = self.last_state.map(|last| (last.0, last.1));
        self.last_state = Some(state);
        self.last_action = Some(action);
        self.last_distance = manhattan_distance(current_pos, game.maze.exit);

        action
    }

    /// Update Q-learning based on the action outcome
    pub fn update_q_learning(&mut self, game: &GameState, moved: bool) {
        if let (Some(state), Some(action)) = (self.last_state, self.last_action) {
            let new_pos = game.player.position;
            let new_state = QState(new_pos.0, new_pos.1, 0);
            let new_distance = manhattan_distance(new_pos, game.maze.exit);
            let reached_exit = game.player.won;
            let backtracked = moved && self.previous_position == Some(new_pos);

            let reward = self.agent.compute_reward(
                moved,
                backtracked,
                new_distance,
                self.last_distance,
                reached_exit,
            );

            self.agent.update_q_value(state, action, reward, new_state, reached_exit);
        }
    }

    pub fn decay_epsilon(&mut self) {
        self.agent.decay_epsilon();
    }

    pub fn set_difficulty(&mut self, difficulty: f32) {
        self.agent.set_difficulty(difficulty);
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::grid::Maze;
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
}

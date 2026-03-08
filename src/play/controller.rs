use crossterm::event::KeyCode;

use crate::core::grid::Direction;

use super::state::{ControlMode, GameState};

pub fn handle_key(game: &mut GameState, code: KeyCode) -> bool {
    match code {
        KeyCode::Esc | KeyCode::Char('q') => true,
        _ if game.player.won => false,
        KeyCode::Char('m') => {
            toggle_mode(game);
            false
        }
        KeyCode::Up | KeyCode::Char('w') => {
            move_pawn(game, Direction::North);
            false
        }
        KeyCode::Right | KeyCode::Char('d') => {
            move_pawn(game, Direction::East);
            false
        }
        KeyCode::Down | KeyCode::Char('s') => {
            move_pawn(game, Direction::South);
            false
        }
        KeyCode::Left | KeyCode::Char('a') => {
            move_pawn(game, Direction::West);
            false
        }
        _ => false,
    }
}

fn move_pawn(game: &mut GameState, direction: Direction) {
    game.start_timer_if_needed();
    if let Some(next) = game.maze.can_move(game.player.position, direction) {
        game.move_player_to(next);
        if game.player.position == game.maze.exit {
            game.mark_won();
        }
    }
}

fn toggle_mode(game: &mut GameState) {
    game.player.control_mode = match game.player.control_mode {
        ControlMode::Manual => {
            game.autosolve_index = nearest_path_index(game);
            ControlMode::AutoSolve
        }
        ControlMode::AutoSolve => ControlMode::MLAgent,
        ControlMode::MLAgent => ControlMode::Manual,
    };
}

fn nearest_path_index(game: &GameState) -> usize {
    let mut best_index = 0;
    let mut best_distance = usize::MAX;

    for (idx, point) in game.autosolve_path.iter().enumerate() {
        let distance = manhattan(*point, game.player.position);
        if distance < best_distance {
            best_distance = distance;
            best_index = idx;
        }
    }

    best_index
}

fn manhattan(a: (usize, usize), b: (usize, usize)) -> usize {
    a.0.abs_diff(b.0) + a.1.abs_diff(b.1)
}

#[cfg(test)]
mod tests {
    use crossterm::event::KeyCode;

    use crate::core::grid::{Direction, Maze};

    use super::handle_key;
    use crate::play::state::{ControlMode, GameState};

    fn corridor_maze() -> Maze {
        let mut maze = Maze::new(2, 1);
        maze.remove_wall_between((0, 0), (1, 0), Direction::East);
        maze
    }

    #[test]
    fn each_keypress_moves_one_step() {
        let maze = corridor_maze();
        let mut game = GameState::new(maze, vec![]);

        handle_key(&mut game, KeyCode::Char('d'));
        assert_eq!(game.player.position, (1, 0));
        assert_eq!(game.player.steps, 1);
        assert_eq!(game.trail, vec![(0, 0)]);
    }

    #[test]
    fn wall_blocks_movement() {
        let maze = Maze::new(2, 1);
        let mut game = GameState::new(maze, vec![]);

        handle_key(&mut game, KeyCode::Char('d'));
        assert_eq!(game.player.position, (0, 0));
        assert_eq!(game.player.steps, 0);
    }

    #[test]
    fn consecutive_keypresses_move_sequentially() {
        let mut maze = Maze::new(4, 1);
        maze.remove_wall_between((0, 0), (1, 0), Direction::East);
        maze.remove_wall_between((1, 0), (2, 0), Direction::East);
        maze.remove_wall_between((2, 0), (3, 0), Direction::East);
        let mut game = GameState::new(maze, vec![]);

        handle_key(&mut game, KeyCode::Char('d'));
        assert_eq!(game.player.position, (1, 0));

        handle_key(&mut game, KeyCode::Char('d'));
        assert_eq!(game.player.position, (2, 0));

        handle_key(&mut game, KeyCode::Char('d'));
        assert_eq!(game.player.position, (3, 0));
    }

    #[test]
    fn toggle_mode_switches_control() {
        let maze = corridor_maze();
        let mut game = GameState::new(maze, vec![(0, 0), (1, 0)]);

        assert_eq!(game.player.control_mode, ControlMode::Manual);
        handle_key(&mut game, KeyCode::Char('m'));
        assert_eq!(game.player.control_mode, ControlMode::AutoSolve);
    }

    #[test]
    fn reaching_exit_marks_won() {
        let maze = corridor_maze();
        let mut game = GameState::new(maze, vec![]);

        handle_key(&mut game, KeyCode::Char('d'));
        assert!(game.player.won);
    }

    #[test]
    fn can_quit_after_winning() {
        let maze = corridor_maze();
        let mut game = GameState::new(maze, vec![]);

        handle_key(&mut game, KeyCode::Char('d'));

        assert!(handle_key(&mut game, KeyCode::Esc));
    }
}

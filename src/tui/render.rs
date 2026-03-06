use std::io::Write;

use crossterm::{
    cursor::MoveTo,
    queue,
    style::Print,
    terminal::{self, Clear, ClearType},
};

use crate::core::grid::{Direction, Maze};
use crate::play::state::{ControlMode, GameState};

/// Draw frame with level context (used in gameplay)
pub fn draw_frame_with_level<W: Write>(
    writer: &mut W,
    game: &GameState,
    level: usize,
    player_wins: usize,
    agent_wins: usize,
    status_message: Option<&str>,
    ml_time_limit_secs: u64,
    ml_step_limit: u64,
) -> std::io::Result<()> {
    let (cols, rows) = terminal::size()?;
    let map = build_map(&game.maze, game);

    let status = fit_status_line(
        status_line_with_level(
        game,
        level,
        player_wins,
        agent_wins,
        status_message,
        ml_time_limit_secs,
        ml_step_limit,
        ),
        cols as usize,
    );
    queue!(writer, MoveTo(0, 0), Clear(ClearType::All), Print(status))?;

    let max_view_h = rows.saturating_sub(1) as usize;
    let max_view_w = cols as usize;

    let player_render = (game.player.position.0 * 2 + 1, game.player.position.1 * 2 + 1);
    let (offset_x, offset_y) = compute_offsets(
        map[0].len(),
        map.len(),
        max_view_w,
        max_view_h,
        player_render,
    );

    for y in 0..max_view_h {
        let map_y = y + offset_y;
        if map_y >= map.len() {
            break;
        }

        let mut line = String::with_capacity(max_view_w);
        for x in 0..max_view_w {
            let map_x = x + offset_x;
            if map_x >= map[map_y].len() {
                break;
            }
            line.push(map[map_y][map_x]);
        }

        queue!(writer, MoveTo(0, (y + 1) as u16), Print(line))?;
    }

    writer.flush()?;
    Ok(())
}

pub fn draw_frame<W: Write>(writer: &mut W, game: &GameState) -> std::io::Result<()> {
    let (cols, rows) = terminal::size()?;
    let map = build_map(&game.maze, game);

    let status = fit_status_line(status_line(game), cols as usize);
    queue!(writer, MoveTo(0, 0), Clear(ClearType::All), Print(status))?;

    let max_view_h = rows.saturating_sub(1) as usize;
    let max_view_w = cols as usize;

    let player_render = (game.player.position.0 * 2 + 1, game.player.position.1 * 2 + 1);
    let (offset_x, offset_y) = compute_offsets(
        map[0].len(),
        map.len(),
        max_view_w,
        max_view_h,
        player_render,
    );

    for y in 0..max_view_h {
        let map_y = y + offset_y;
        if map_y >= map.len() {
            break;
        }

        let mut line = String::with_capacity(max_view_w);
        for x in 0..max_view_w {
            let map_x = x + offset_x;
            if map_x >= map[map_y].len() {
                break;
            }
            line.push(map[map_y][map_x]);
        }

        queue!(writer, MoveTo(0, (y + 1) as u16), Print(line))?;
    }

    writer.flush()?;
    Ok(())
}

fn status_line(game: &GameState) -> String {
    let mode = match game.player.control_mode {
        ControlMode::Manual => "MAN",
        ControlMode::AutoSolve => "AUTO",
        ControlMode::MLAgent => "ML",
    };

    let won = if game.player.won { " | WIN" } else { "" };
    format!(
        "Q quit | M mode | {} | {} st | {}s{}",
        mode,
        game.player.steps,
        game.elapsed_secs(),
        won
    )
}

fn status_line_with_level(
    game: &GameState,
    level: usize,
    player_wins: usize,
    agent_wins: usize,
    status_message: Option<&str>,
    ml_time_limit_secs: u64,
    ml_step_limit: u64,
) -> String {
    let mode = match game.player.control_mode {
        ControlMode::Manual => "MAN",
        ControlMode::AutoSolve => "AUTO",
        ControlMode::MLAgent => "ML",
    };

    let won_status = if game.player.won {
        if game.player.control_mode == ControlMode::MLAgent {
            " | WIN auto"
        } else {
            " | WIN N-next"
        }
    } else {
        ""
    };
    let ml_limits = if game.player.control_mode == ControlMode::MLAgent && !game.player.won {
        format!(
            " | ml {}/{} st {}/{}s",
            game.player.steps,
            ml_step_limit,
            game.elapsed_secs(),
            ml_time_limit_secs
        )
    } else {
        String::new()
    };
    let message = status_message
        .map(short_status_message)
        .map(|msg| format!(" | {}", msg))
        .unwrap_or_default();

    format!(
        "Q quit | M mode | L{} {} | {} st | {}s | P:{} A:{}{}{}{}",
        level,
        mode,
        game.player.steps,
        game.elapsed_secs(),
        player_wins,
        agent_wins,
        won_status,
        ml_limits,
        message
    )
}

fn fit_status_line(mut status: String, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }

    let count = status.chars().count();
    if count <= max_width {
        return status;
    }

    if max_width <= 3 {
        return ".".repeat(max_width);
    }

    status = status.chars().take(max_width - 3).collect();
    status.push_str("...");
    status
}

fn short_status_message(message: &str) -> String {
    if message.contains("time limit") {
        "reset:time".to_string()
    } else if message.contains("step limit") {
        "reset:steps".to_string()
    } else {
        message.to_string()
    }
}

fn compute_offsets(
    map_w: usize,
    map_h: usize,
    view_w: usize,
    view_h: usize,
    focus: (usize, usize),
) -> (usize, usize) {
    let mut offset_x = focus.0.saturating_sub(view_w / 2);
    let mut offset_y = focus.1.saturating_sub(view_h / 2);

    if offset_x + view_w > map_w {
        offset_x = map_w.saturating_sub(view_w);
    }
    if offset_y + view_h > map_h {
        offset_y = map_h.saturating_sub(view_h);
    }

    (offset_x, offset_y)
}

fn build_map(maze: &Maze, game: &GameState) -> Vec<Vec<char>> {
    let map_w = maze.width() * 2 + 1;
    let map_h = maze.height() * 2 + 1;
    let mut map = vec![vec!['█'; map_w]; map_h];

    for y in 0..maze.height() {
        for x in 0..maze.width() {
            let render_x = x * 2 + 1;
            let render_y = y * 2 + 1;
            map[render_y][render_x] = ' ';

            let cell = maze.cell(x, y);
            if !cell.has_wall(Direction::East) {
                map[render_y][render_x + 1] = ' ';
            }
            if !cell.has_wall(Direction::South) {
                map[render_y + 1][render_x] = ' ';
            }
        }
    }

    if game.player.control_mode == ControlMode::AutoSolve {
        for point in &game.autosolve_path {
            let x = point.0 * 2 + 1;
            let y = point.1 * 2 + 1;
            map[y][x] = '.';
        }
    }

    let start = (maze.start.0 * 2 + 1, maze.start.1 * 2 + 1);
    let exit = (maze.exit.0 * 2 + 1, maze.exit.1 * 2 + 1);
    map[start.1][start.0] = 'S';
    map[exit.1][exit.0] = 'E';

    let pawn = (game.player.position.0 * 2 + 1, game.player.position.1 * 2 + 1);
    map[pawn.1][pawn.0] = '@';

    map
}

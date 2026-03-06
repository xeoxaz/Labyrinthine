use std::io::{stdout, Write};
use std::thread;
use std::time::{Duration, Instant};

use crossterm::cursor::{Hide, Show};
use crossterm::event::{self, Event, KeyEventKind};
use crossterm::terminal::{
    self, disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::execute;
use rand::Rng;
use rand::SeedableRng;

mod core;
mod play;
mod runtime;
mod tui;
mod ml;

use core::generator::generate_recursive_backtracker;
use play::controller;
use play::state::{GameState, ControlMode};
use play::level_game::LevelGame;
use runtime::resources::{apply as apply_resource_policy, ResourcePolicy};
use tui::render;

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let command = parse_command(&args);

    apply_resource_policy(ResourcePolicy {
        max_mode: command.max_mode,
        threads: command.threads,
    })?;

    match command.mode.as_str() {
        "generate" => run_generate(&command),
        "play" => run_play(&command),
        _ => Err(format!("unknown mode: {}", command.mode)),
    }
}

fn run_generate(command: &Command) -> Result<(), String> {
    let seed = rand::rng().random::<u64>();
    let maze = generate_recursive_backtracker(command.width, command.height, seed);
    let game = GameState::new(maze, Vec::new());
    let map = render_for_stdout(&game);
    println!("{map}");
    Ok(())
}

fn run_play(_command: &Command) -> Result<(), String> {
    let (cols, rows) = terminal::size().map_err(|e| e.to_string())?;
    let max_width = ((cols as usize).saturating_sub(1) / 2).max(2);
    let max_height = ((rows as usize).saturating_sub(2) / 2).max(2);
    let session_seed = rand::rng().random::<u64>();
    let mut level_game = LevelGame::new(session_seed, max_width, max_height);
    let mut rng = rand::rngs::StdRng::seed_from_u64(session_seed);
    let mut ml_auto_advance_at: Option<Instant> = None;

    enable_raw_mode().map_err(|e| e.to_string())?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen, Hide).map_err(|e| e.to_string())?;
    let _guard = TerminalGuard;

    let tick = Duration::from_millis(33);

    loop {
        let frame_start = Instant::now();

        // Handle keyboard input
        while event::poll(Duration::from_millis(1)).map_err(|e| e.to_string())? {
            if let Event::Key(key) = event::read().map_err(|e| e.to_string())? {
                if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
                    // Handle 'N' for next level (player won current level)
                    if key.code == crossterm::event::KeyCode::Char('n') && level_game.current_game.player.won {
                        level_game.level_manager.record_player_win();
                        level_game.next_level();
                        ml_auto_advance_at = None;
                        continue;
                    }

                    if controller::handle_key(&mut level_game.current_game, key.code) {
                        return Ok(());
                    }
                }
            }
        }

        if level_game.current_game.player.control_mode == ControlMode::MLAgent
            && level_game.current_game.player.won
        {
            if let Some(deadline) = ml_auto_advance_at {
                if Instant::now() >= deadline {
                    level_game.next_level();
                    ml_auto_advance_at = None;
                    continue;
                }
            } else {
                ml_auto_advance_at = Some(Instant::now() + Duration::from_millis(900));
            }
        } else {
            ml_auto_advance_at = None;
        }

        // ML agent tick (automatic moves)
        if level_game.current_game.player.control_mode == ControlMode::MLAgent
            && !level_game.current_game.player.won
        {
            if level_game.ml_solver.is_none() {
                level_game.create_ml_solver();
            }

            level_game.current_game.start_timer_if_needed();
            let time_limit_secs = level_game.ml_time_limit_secs();
            let step_limit = level_game.ml_step_limit();
            let mut failure_reason = None;

            if let Some(solver) = &mut level_game.ml_solver {
                let action = solver.step(&level_game.current_game, &mut rng);
                let direction = action.to_direction();
                let old_pos = level_game.current_game.player.position;

                // Try to move
                if let Some(next) = level_game.current_game.maze.can_move(old_pos, direction) {
                    level_game.current_game.player.position = next;
                    level_game.current_game.player.steps += 1;
                }

                // Check win
                let moved = level_game.current_game.player.position != old_pos;
                if level_game.current_game.player.position == level_game.current_game.maze.exit {
                    level_game.current_game.mark_won();
                    level_game.level_manager.record_agent_win();
                }

                // Update Q-learning
                solver.update_q_learning(&level_game.current_game, moved);
                solver.decay_epsilon();

                if !level_game.current_game.player.won {
                    if level_game.current_game.elapsed_secs() >= time_limit_secs {
                        solver.record_failure(&level_game.current_game);
                        failure_reason = Some(format!(
                            "ML reset: hit {}s time limit",
                            time_limit_secs
                        ));
                    } else if level_game.current_game.player.steps >= step_limit {
                        solver.record_failure(&level_game.current_game);
                        failure_reason = Some(format!(
                            "ML reset: hit {} step limit",
                            step_limit
                        ));
                    }
                }
            }

            if let Some(reason) = failure_reason {
                level_game.restart_ml_attempt(reason);
                ml_auto_advance_at = None;
            }
        }

        // Render frame
        render::draw_frame_with_level(
            &mut stdout,
            &level_game.current_game,
            level_game.level_manager.current_level(),
            level_game.level_manager.player_wins(),
            level_game.level_manager.agent_wins(),
            level_game.status_message.as_deref(),
            level_game.ml_time_limit_secs(),
            level_game.ml_step_limit(),
        ).map_err(|e| e.to_string())?;

        let elapsed = frame_start.elapsed();
        if elapsed < tick {
            thread::sleep(tick - elapsed);
        }
    }
}

fn render_for_stdout(game: &GameState) -> String {
    let map_w = game.maze.width() * 2 + 1;
    let map_h = game.maze.height() * 2 + 1;
    let mut map = vec![vec!['#'; map_w]; map_h];

    for y in 0..game.maze.height() {
        for x in 0..game.maze.width() {
            let rx = x * 2 + 1;
            let ry = y * 2 + 1;
            map[ry][rx] = ' ';

            let cell = game.maze.cell(x, y);
            if !cell.has_wall(core::grid::Direction::East) {
                map[ry][rx + 1] = ' ';
            }
            if !cell.has_wall(core::grid::Direction::South) {
                map[ry + 1][rx] = ' ';
            }
        }
    }

    let pawn = game.player.position;
    map[pawn.1 * 2 + 1][pawn.0 * 2 + 1] = '@';
    map[game.maze.start.1 * 2 + 1][game.maze.start.0 * 2 + 1] = 'S';
    map[game.maze.exit.1 * 2 + 1][game.maze.exit.0 * 2 + 1] = 'E';

    map.into_iter()
        .map(|row| row.into_iter().collect::<String>())
        .collect::<Vec<String>>()
        .join("\n")
}

#[derive(Debug)]
struct Command {
    mode: String,
    width: usize,
    height: usize,
    max_mode: bool,
    threads: Option<usize>,
}

fn parse_command(args: &[String]) -> Command {
    let mode = args.get(1).cloned().unwrap_or_else(|| "play".to_string());
    let width = parse_usize_flag(args, "--width").unwrap_or(39).max(2);
    let height = parse_usize_flag(args, "--height").unwrap_or(21).max(2);
    let max_mode = args.iter().any(|arg| arg == "--max-mode");
    let threads = parse_usize_flag(args, "--threads");

    Command {
        mode,
        width,
        height,
        max_mode,
        threads,
    }
}

fn parse_usize_flag(args: &[String], flag: &str) -> Option<usize> {
    value_after(args, flag)?.parse().ok()
}

fn value_after<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|window| window[0] == flag)
        .map(|window| window[1].as_str())
}

struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let mut out = stdout();
        let _ = execute!(out, Show, LeaveAlternateScreen);
        let _ = out.flush();
    }
}

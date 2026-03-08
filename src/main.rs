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
use runtime::gpu::{detect_gpu_probe, resolve_ml_runtime, GpuPolicy};
use runtime::resources::{apply as apply_resource_policy, resolve_thread_count, ResourcePolicy};
use tui::render;

const SIMULATION_DT: Duration = Duration::from_millis(8);
const RENDER_DT: Duration = Duration::from_millis(83);
const AUTO_ADVANCE_DELAY: Duration = Duration::from_millis(900);
const MAX_SIMULATION_STEPS: usize = 8;
const LOADING_BATCH_STEPS: usize = 48;
const RELOAD_SCREEN_DURATION: Duration = Duration::from_millis(850);
const BOOT_SCREEN_DURATION: Duration = Duration::from_millis(2200);
const BOOT_POLL_INTERVAL: Duration = Duration::from_millis(40);
const IDLE_SLEEP: Duration = Duration::from_millis(1);

struct ViewCycleState {
    mode: render::LevelViewMode,
    next_switch_at: Option<Instant>,
}

impl ViewCycleState {
    fn new() -> Self {
        Self {
            mode: render::LevelViewMode::Normal,
            next_switch_at: None,
        }
    }

    fn reset(&mut self) -> bool {
        let changed = self.mode != render::LevelViewMode::Normal || self.next_switch_at.is_some();
        self.mode = render::LevelViewMode::Normal;
        self.next_switch_at = None;
        changed
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let command = match parse_command(&args)? {
        CommandAction::Run(command) => command,
        CommandAction::PrintHelp => {
            print!("{}", usage_text(&args));
            return Ok(());
        }
        CommandAction::PrintVersion => {
            println!("labyrinthine {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
    };

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
    let seed = command.seed.unwrap_or_else(|| rand::rng().random::<u64>());
    let maze = generate_recursive_backtracker(
        command.width.unwrap_or(39).max(2),
        command.height.unwrap_or(21).max(2),
        seed,
    );
    let game = GameState::new(maze, Vec::new());
    let map = render_for_stdout(&game);
    println!("{map}");
    Ok(())
}

fn run_play(_command: &Command) -> Result<(), String> {
    let gpu_policy = GpuPolicy {
        prefer_gpu: _command.gpu_mode,
        require_gpu: _command.require_gpu,
    };
    let (cols, rows) = terminal::size().map_err(|e| e.to_string())?;
    let (max_width, max_height) = resolve_play_dimensions(_command, cols, rows);
    let session_seed = _command.seed.unwrap_or_else(|| rand::rng().random::<u64>());
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let configured_threads = resolve_thread_count(ResourcePolicy {
        max_mode: _command.max_mode,
        threads: _command.threads,
    });
    let gpu_probe = detect_gpu_probe();
    let ml_runtime = resolve_ml_runtime(gpu_policy, gpu_probe)?;
    let mut level_game = LevelGame::new(session_seed, max_width, max_height);
    let mut rng = rand::rngs::StdRng::seed_from_u64(session_seed);
    let mut ml_auto_advance_at: Option<Instant> = None;
    let mut ml_reload_at: Option<Instant> = None;
    let mut view_cycle = ViewCycleState::new();
    level_game.status_message = Some(ml_runtime.detail_message().to_string());

    enable_raw_mode().map_err(|e| e.to_string())?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen, Hide).map_err(|e| e.to_string())?;
    let _guard = TerminalGuard;

    if run_boot_sequence(
        &mut stdout,
        render::BootDiagnostics {
            terminal_cols: cols,
            terminal_rows: rows,
            available_threads: Some(available_threads),
            configured_threads: Some(configured_threads),
            cuda_available: None,
            rocm_available: None,
            vulkan_available: None,
            runtime_label: None,
            runtime_detail: None,
            session_seed,
        },
        gpu_probe,
        ml_runtime.short_label().to_string(),
        ml_runtime.detail_message().to_string(),
    )? {
        return Ok(());
    }

    if level_game.current_game.player.control_mode == ControlMode::MLAgent {
        level_game.begin_initial_loading();
    }

    let mut last_simulation_tick = Instant::now();
    let mut accumulated = Duration::ZERO;
    let mut next_render_at = last_simulation_tick;
    let mut dirty = true;

    loop {
        match process_input(&mut level_game, &mut ml_auto_advance_at, &mut ml_reload_at)? {
            InputResult::Continue(changed) => dirty |= changed,
            InputResult::Quit => return Ok(()),
        }

        let now = Instant::now();
        accumulated += now.saturating_duration_since(last_simulation_tick);
        last_simulation_tick = now;
        dirty |= update_view_cycle(&level_game, &mut view_cycle, &mut rng, now);

        let mut sim_steps = 0;
        while accumulated >= SIMULATION_DT && sim_steps < MAX_SIMULATION_STEPS {
            dirty |= advance_simulation(
                &mut level_game,
                &mut rng,
                &mut ml_auto_advance_at,
                &mut ml_reload_at,
            );
            accumulated = accumulated.saturating_sub(SIMULATION_DT);
            sim_steps += 1;
        }

        if sim_steps == MAX_SIMULATION_STEPS {
            accumulated = Duration::ZERO;
        }

        let now = Instant::now();
        if dirty || now >= next_render_at {
            if let Some(loading) = level_game.loading_level.as_ref() {
                render::draw_loading_screen(
                    &mut stdout,
                    loading,
                    ml_runtime.short_label(),
                    level_game.ml_hud_stats(),
                    level_game.maze_seed,
                ).map_err(|e| e.to_string())?;
            } else if let Some(reload) = level_game.reload_state.as_ref() {
                render::draw_reload_screen(
                    &mut stdout,
                    reload,
                    ml_runtime.short_label(),
                    level_game.maze_seed,
                ).map_err(|e| e.to_string())?;
            } else {
                render::draw_frame_with_level(
                    &mut stdout,
                    &level_game.current_game,
                    level_game.level_manager.current_level(),
                    level_game.level_manager.player_wins(),
                    level_game.level_manager.agent_wins(),
                    ml_runtime.short_label(),
                    level_game.status_message.as_deref(),
                    level_game.ml_time_limit_secs(),
                    level_game.ml_step_limit(),
                    level_game.ml_hud_stats(),
                    view_cycle.mode,
                    level_game.maze_seed,
                ).map_err(|e| e.to_string())?;
            }
            next_render_at = now + RENDER_DT;
            dirty = false;
            continue;
        }

        let until_simulation = SIMULATION_DT.saturating_sub(accumulated);
        let until_render = next_render_at.saturating_duration_since(now);
        let sleep_for = until_simulation.min(until_render).min(IDLE_SLEEP);
        if !sleep_for.is_zero() {
            thread::sleep(sleep_for);
        }
    }
}

fn update_view_cycle(
    level_game: &LevelGame,
    view_cycle: &mut ViewCycleState,
    rng: &mut rand::rngs::StdRng,
    now: Instant,
) -> bool {
    let ml_active = level_game.loading_level.is_none()
        && level_game.current_game.player.control_mode == ControlMode::MLAgent
        && !level_game.current_game.player.won;

    if !ml_active {
        return view_cycle.reset();
    }

    if view_cycle.next_switch_at.is_none() {
        view_cycle.next_switch_at = Some(now + next_view_switch_delay(rng));
        return false;
    }

    if let Some(deadline) = view_cycle.next_switch_at {
        if now >= deadline {
            view_cycle.mode = match view_cycle.mode {
                render::LevelViewMode::Normal => render::LevelViewMode::AgentFocus,
                render::LevelViewMode::AgentFocus => render::LevelViewMode::Normal,
            };
            view_cycle.next_switch_at = Some(now + next_view_switch_delay(rng));
            return true;
        }
    }

    false
}

fn next_view_switch_delay(rng: &mut rand::rngs::StdRng) -> Duration {
    Duration::from_secs(rng.random_range(5..=10))
}

fn run_boot_sequence<W: Write>(
    writer: &mut W,
    mut diagnostics: render::BootDiagnostics,
    gpu_probe: runtime::gpu::GpuProbe,
    runtime_label: String,
    runtime_detail: String,
) -> Result<bool, String> {
    let started_at = Instant::now();
    let deadline = started_at + BOOT_SCREEN_DURATION;
    loop {
        let now = Instant::now();
        let elapsed = now.saturating_duration_since(started_at);
        let progress_ratio = (elapsed.as_secs_f32() / BOOT_SCREEN_DURATION.as_secs_f32()).clamp(0.0, 1.0);
        let stage = ((progress_ratio * 6.0).floor() as usize).min(5);
        let pulse = ((elapsed.as_millis() / 90) % 4) as usize;

        if stage >= 1 {
            diagnostics.cuda_available = Some(gpu_probe.cuda);
        }
        if stage >= 2 {
            diagnostics.rocm_available = Some(gpu_probe.rocm);
        }
        if stage >= 3 {
            diagnostics.vulkan_available = Some(gpu_probe.vulkan);
        }
        if stage >= 4 {
            diagnostics.runtime_label = Some(runtime_label.clone());
            diagnostics.runtime_detail = Some(runtime_detail.clone());
        }

        render::draw_boot_screen(
            writer,
            &diagnostics,
            progress_ratio,
            pulse,
        )
        .map_err(|e| e.to_string())?;

        if event::poll(BOOT_POLL_INTERVAL).map_err(|e| e.to_string())? {
            if let Event::Key(key) = event::read().map_err(|e| e.to_string())? {
                if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
                    return Ok(matches!(key.code, crossterm::event::KeyCode::Esc | crossterm::event::KeyCode::Char('q')));
                }
            }
        }

        if now >= deadline {
            return Ok(false);
        }
    }
}

fn process_input(
    level_game: &mut LevelGame,
    ml_auto_advance_at: &mut Option<Instant>,
    ml_reload_at: &mut Option<Instant>,
) -> Result<InputResult, String> {
    let mut dirty = false;

    while event::poll(Duration::ZERO).map_err(|e| e.to_string())? {
        if let Event::Key(key) = event::read().map_err(|e| e.to_string())? {
            if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
                if level_game.loading_level.is_some() || level_game.reload_state.is_some() {
                    if matches!(key.code, crossterm::event::KeyCode::Esc | crossterm::event::KeyCode::Char('q')) {
                        return Ok(InputResult::Quit);
                    }
                    dirty = true;
                    continue;
                }

                if key.code == crossterm::event::KeyCode::Char('n')
                    && level_game.current_game.player.won
                {
                    if level_game.current_game.player.control_mode == ControlMode::MLAgent {
                        level_game.begin_next_level_loading();
                    } else {
                        level_game.level_manager.record_player_win();
                        level_game.next_level();
                    }
                    *ml_auto_advance_at = None;
                    *ml_reload_at = None;
                    dirty = true;
                    continue;
                }

                if controller::handle_key(&mut level_game.current_game, key.code) {
                    return Ok(InputResult::Quit);
                }
                dirty = true;
            }
        }
    }

    Ok(InputResult::Continue(dirty))
}

enum InputResult {
    Continue(bool),
    Quit,
}

fn advance_simulation(
    level_game: &mut LevelGame,
    rng: &mut rand::rngs::StdRng,
    ml_auto_advance_at: &mut Option<Instant>,
    ml_reload_at: &mut Option<Instant>,
) -> bool {
    if level_game.reload_state.is_some() {
        return advance_reload(level_game, ml_reload_at);
    }

    if level_game.loading_level.is_some() {
        return advance_loading(level_game, rng);
    }

    if level_game.current_game.player.control_mode == ControlMode::MLAgent
        && level_game.current_game.player.won
    {
        if let Some(deadline) = ml_auto_advance_at {
            if Instant::now() >= *deadline {
                level_game.begin_next_level_loading();
                *ml_auto_advance_at = None;
                return true;
            }
        } else {
            *ml_auto_advance_at = Some(Instant::now() + AUTO_ADVANCE_DELAY);
            return true;
        }
        return false;
    }

    *ml_auto_advance_at = None;

    if level_game.current_game.player.control_mode != ControlMode::MLAgent
        || level_game.current_game.player.won
    {
        return false;
    }

    if level_game.ml_solver.is_none() {
        level_game.create_ml_solver();
    }

    level_game.current_game.start_timer_if_needed();
    let time_limit_secs = level_game.ml_time_limit_secs();
    let step_limit = level_game.ml_step_limit();
    let mut failure_reason = None;

    if let Some(solver) = &mut level_game.ml_solver {
        let action = solver.step(&level_game.current_game, rng);
        let direction = action.to_direction();
        let old_pos = level_game.current_game.player.position;

        if let Some(next) = level_game.current_game.maze.can_move(old_pos, direction) {
            level_game.current_game.move_player_to(next);
        }

        let moved = level_game.current_game.player.position != old_pos;
        if level_game.current_game.player.position == level_game.current_game.maze.exit {
            level_game.current_game.mark_won();
            level_game.level_manager.record_agent_win();
            solver.record_win(level_game.current_game.player.steps);
            solver.decay_epsilon();
        }

        solver.update_q_learning(&level_game.current_game, moved);

        if !level_game.current_game.player.won {
            if level_game.current_game.elapsed_secs() >= time_limit_secs {
                solver.record_failure(&level_game.current_game);
                solver.decay_epsilon();
                failure_reason = Some(format!("ML reset: hit {}s time limit", time_limit_secs));
            } else if level_game.current_game.player.steps >= step_limit {
                solver.record_failure(&level_game.current_game);
                solver.decay_epsilon();
                failure_reason = Some(format!("ML reset: hit {} step limit", step_limit));
            }
        }
    }

    if let Some(reason) = failure_reason {
        level_game.begin_ml_reload(reason);
        *ml_auto_advance_at = None;
        *ml_reload_at = None;
    }

    true
}

fn advance_reload(level_game: &mut LevelGame, ml_reload_at: &mut Option<Instant>) -> bool {
    if level_game.reload_state.is_none() {
        *ml_reload_at = None;
        return false;
    }

    if let Some(deadline) = ml_reload_at {
        if Instant::now() >= *deadline {
            level_game.finish_ml_reload();
            *ml_reload_at = None;
        }
    } else {
        *ml_reload_at = Some(Instant::now() + RELOAD_SCREEN_DURATION);
    }

    true
}

fn advance_loading(level_game: &mut LevelGame, rng: &mut rand::rngs::StdRng) -> bool {
    if level_game.ml_solver.is_none() {
        level_game.create_ml_solver();
    }

    let mut loading = match level_game.loading_level.take() {
        Some(loading) => loading,
        None => return false,
    };

    let time_limit_secs = level_game
        .level_manager
        .ml_time_limit_secs(&loading.training_game.maze, loading.training_game.autosolve_path.len());
    let step_limit = level_game
        .level_manager
        .ml_step_limit(&loading.training_game.maze, loading.training_game.autosolve_path.len());
    let mut ready = false;

    if let Some(solver) = &mut level_game.ml_solver {
        for _ in 0..LOADING_BATCH_STEPS {
            match advance_ml_agent_step(&mut loading.training_game, solver, rng, time_limit_secs, step_limit) {
                MlStepState::Running => {}
                MlStepState::Won => {
                    loading.warmup_episodes_done += 1;
                    loading.last_summary = format!(
                        "Warmup run {} solved the maze",
                        loading.warmup_episodes_done
                    );
                    if loading.warmup_episodes_done >= loading.warmup_episodes_total {
                        ready = true;
                        break;
                    }
                    solver.begin_new_episode();
                    loading.training_game = loading.template_game.clone();
                }
                MlStepState::Failed(reason) => {
                    loading.warmup_episodes_done += 1;
                    loading.last_summary = reason;
                    if loading.warmup_episodes_done >= loading.warmup_episodes_total {
                        ready = true;
                        break;
                    }
                    solver.begin_new_episode();
                    loading.training_game = loading.template_game.clone();
                }
            }
        }
    }

    if ready {
        level_game.finish_loading_level(loading);
    } else {
        level_game.loading_level = Some(loading);
    }

    true
}

enum MlStepState {
    Running,
    Won,
    Failed(String),
}

fn advance_ml_agent_step(
    game: &mut GameState,
    solver: &mut play::ml_solver::MLSolver,
    rng: &mut rand::rngs::StdRng,
    time_limit_secs: u64,
    step_limit: u64,
) -> MlStepState {
    game.start_timer_if_needed();
    let action = solver.step(game, rng);
    let direction = action.to_direction();
    let old_pos = game.player.position;

    if let Some(next) = game.maze.can_move(old_pos, direction) {
        game.move_player_to(next);
    }

    let moved = game.player.position != old_pos;
    if game.player.position == game.maze.exit {
        game.mark_won();
        solver.record_win(game.player.steps);
        solver.decay_epsilon();
        solver.update_q_learning(game, moved);
        return MlStepState::Won;
    }

    solver.update_q_learning(game, moved);

    if game.elapsed_secs() >= time_limit_secs {
        solver.record_failure(game);
        solver.decay_epsilon();
        return MlStepState::Failed(format!(
            "Warmup reset after reaching the {} second time limit",
            time_limit_secs
        ));
    }

    if game.player.steps >= step_limit {
        solver.record_failure(game);
        solver.decay_epsilon();
        return MlStepState::Failed(format!(
            "Warmup reset after reaching the {} step limit",
            step_limit
        ));
    }

    MlStepState::Running
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
    width: Option<usize>,
    height: Option<usize>,
    seed: Option<u64>,
    max_mode: bool,
    threads: Option<usize>,
    gpu_mode: bool,
    require_gpu: bool,
}

enum CommandAction {
    Run(Command),
    PrintHelp,
    PrintVersion,
}

fn parse_command(args: &[String]) -> Result<CommandAction, String> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") || args.get(1).is_some_and(|arg| arg == "help") {
        return Ok(CommandAction::PrintHelp);
    }

    if args.iter().any(|arg| arg == "--version" || arg == "-V") {
        return Ok(CommandAction::PrintVersion);
    }

    let mode = args.get(1).cloned().unwrap_or_else(|| "play".to_string());
    if mode != "play" && mode != "generate" {
        return Err(format!("unknown mode: {mode}"));
    }

    let width = parse_usize_flag(args, "--width").map(|value| value.max(2));
    let height = parse_usize_flag(args, "--height").map(|value| value.max(2));
    let seed = parse_u64_flag(args, "--seed");
    let max_mode = args.iter().any(|arg| arg == "--max-mode");
    let threads = parse_usize_flag(args, "--threads");
    let cpu_only = args.iter().any(|arg| arg == "--cpu-only");
    let gpu_mode = args.iter().any(|arg| arg == "--gpu-mode") || !cpu_only;
    let require_gpu = args.iter().any(|arg| arg == "--require-gpu");

    Ok(CommandAction::Run(Command {
        mode,
        width,
        height,
        seed,
        max_mode,
        threads,
        gpu_mode,
        require_gpu,
    }))
}

fn parse_usize_flag(args: &[String], flag: &str) -> Option<usize> {
    value_after(args, flag)?.parse().ok()
}

fn parse_u64_flag(args: &[String], flag: &str) -> Option<u64> {
    value_after(args, flag)?.parse().ok()
}

fn resolve_play_dimensions(command: &Command, cols: u16, rows: u16) -> (usize, usize) {
    let terminal_max_width = ((cols as usize).saturating_sub(1) / 2).max(2);
    let terminal_max_height = ((rows as usize).saturating_sub(2) / 2).max(2);

    let width = command
        .width
        .unwrap_or(terminal_max_width)
        .min(terminal_max_width)
        .max(2);
    let height = command
        .height
        .unwrap_or(terminal_max_height)
        .min(terminal_max_height)
        .max(2);

    (width, height)
}

fn usage_text(args: &[String]) -> String {
    let bin = args.first().map(String::as_str).unwrap_or("labyrinthine");
    format!(
        "Labyrinthine {}\n\nUsage:\n  {bin} [play] [options]\n  {bin} generate [options]\n  {bin} --help\n  {bin} --version\n\nModes:\n  play      Start the terminal maze runner (default)\n  generate  Print a generated maze to stdout\n\nOptions:\n  --width N       Maze width; in play mode, capped by terminal size\n  --height N      Maze height; in play mode, capped by terminal size\n  --seed N        Use a deterministic maze/session seed\n  --max-mode      Use all available CPU threads\n  --threads N     Set an explicit worker thread count\n  --gpu-mode      Prefer GPU-capable ML runtime when probe tools are available\n  --cpu-only      Disable GPU probing and force CPU runtime\n  --require-gpu   Fail if no supported GPU backend is detected\n  -h, --help      Show this help text\n  -V, --version   Show version information\n",
        env!("CARGO_PKG_VERSION")
    )
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

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn parse_help_flag_returns_help_action() {
        let args = vec!["labyrinthine".to_string(), "--help".to_string()];

        assert!(matches!(parse_command(&args).unwrap(), CommandAction::PrintHelp));
    }

    #[test]
    fn parse_version_flag_returns_version_action() {
        let args = vec!["labyrinthine".to_string(), "--version".to_string()];

        assert!(matches!(parse_command(&args).unwrap(), CommandAction::PrintVersion));
    }

    #[test]
    fn parse_run_command_captures_seed_and_dimensions() {
        let args = vec![
            "labyrinthine".to_string(),
            "generate".to_string(),
            "--width".to_string(),
            "25".to_string(),
            "--height".to_string(),
            "15".to_string(),
            "--seed".to_string(),
            "42".to_string(),
        ];

        let CommandAction::Run(command) = parse_command(&args).unwrap() else {
            panic!("expected run command");
        };

        assert_eq!(command.mode, "generate");
        assert_eq!(command.width, Some(25));
        assert_eq!(command.height, Some(15));
        assert_eq!(command.seed, Some(42));
    }

    #[test]
    fn play_dimensions_default_to_terminal_bounds() {
        let command = Command {
            mode: "play".to_string(),
            width: None,
            height: None,
            seed: None,
            max_mode: false,
            threads: None,
            gpu_mode: true,
            require_gpu: false,
        };

        assert_eq!(resolve_play_dimensions(&command, 120, 40), (59, 19));
    }

    #[test]
    fn play_dimensions_respect_requested_caps() {
        let command = Command {
            mode: "play".to_string(),
            width: Some(30),
            height: Some(10),
            seed: Some(7),
            max_mode: false,
            threads: None,
            gpu_mode: true,
            require_gpu: false,
        };

        assert_eq!(resolve_play_dimensions(&command, 120, 40), (30, 10));
        assert_eq!(resolve_play_dimensions(&command, 40, 12), (19, 5));
    }
}

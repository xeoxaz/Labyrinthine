use std::io::Write;

use crossterm::{
    cursor::MoveTo,
    queue,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{self, Clear, ClearType},
};

use crate::core::grid::{Direction, Maze};
use crate::play::level_game::LevelLoadingState;
use crate::play::ml_solver::MlHudStats;
use crate::play::state::{ControlMode, GameState};

const AGENT_FOCUS_RADIUS_X: usize = 2;
const AGENT_FOCUS_RADIUS_Y: usize = 2;

#[derive(Clone, Debug)]
pub struct BootDiagnostics {
    pub terminal_cols: u16,
    pub terminal_rows: u16,
    pub available_threads: Option<usize>,
    pub configured_threads: Option<usize>,
    pub cuda_available: Option<bool>,
    pub rocm_available: Option<bool>,
    pub vulkan_available: Option<bool>,
    pub runtime_label: Option<String>,
    pub runtime_detail: Option<String>,
    pub session_seed: u64,
}

#[derive(Clone, Copy)]
struct Tile {
    glyph: char,
    color: Color,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelViewMode {
    Normal,
    AgentFocus,
}

/// Draw frame with level context (used in gameplay)
pub fn draw_frame_with_level<W: Write>(
    writer: &mut W,
    game: &GameState,
    level: usize,
    player_wins: usize,
    agent_wins: usize,
    ml_runtime_label: &str,
    status_message: Option<&str>,
    ml_time_limit_secs: u64,
    ml_step_limit: u64,
    ml_stats: Option<MlHudStats>,
    view_mode: LevelViewMode,
) -> std::io::Result<()> {
    let (cols, rows) = terminal::size()?;
    let map = build_map(&game.maze, game, view_mode);
    queue!(writer, MoveTo(0, 0), Clear(ClearType::All))?;

    let hud = hud_lines_with_level(
        game,
        level,
        player_wins,
        agent_wins,
        ml_runtime_label,
        status_message,
        ml_time_limit_secs,
        ml_step_limit,
        ml_stats,
        view_mode,
        cols as usize,
    );
    render_centered_text(writer, 0, cols, &hud.0, hud_primary_color(game))?;

    let view_h = rows.saturating_sub(2) as usize;
    let view_w = cols as usize;

    let player_render = (game.player.position.0 * 2 + 1, game.player.position.1 * 2 + 1);
    let viewport = compute_viewport(
        map[0].len(),
        map.len(),
        view_w,
        view_h,
        player_render,
    );

    for y in 0..viewport.visible_h {
        let map_y = y + viewport.offset_y;

        let mut row_tiles = Vec::with_capacity(viewport.visible_w);
        for x in 0..viewport.visible_w {
            let map_x = x + viewport.offset_x;
            row_tiles.push(map[map_y][map_x]);
        }

        render_tile_row(
            writer,
            viewport.origin_x as u16,
            (1 + viewport.origin_y + y) as u16,
            &row_tiles,
        )?;
    }

    if rows > 1 {
        render_centered_text(writer, rows - 1, cols, &hud.1, hud_secondary_color(game))?;
    }

    queue!(writer, ResetColor)?;
    writer.flush()?;
    Ok(())
}

pub fn draw_loading_screen<W: Write>(
    writer: &mut W,
    loading: &LevelLoadingState,
    ml_runtime_label: &str,
    ml_stats: Option<MlHudStats>,
) -> std::io::Result<()> {
    let (cols, rows) = terminal::size()?;
    queue!(writer, MoveTo(0, 0), Clear(ClearType::All))?;

    let title = format!("Loading Level {:02}", loading.level);
    let subtitle = "ML agent is learning the new maze before play begins";
    let progress = progress_bar(loading.progress_ratio(), (cols as usize).saturating_sub(20).clamp(16, 40));
    let detail = format!(
        "Warmup run {} of {}",
        loading.warmup_episodes_done.saturating_add(1).min(loading.warmup_episodes_total),
        loading.warmup_episodes_total,
    );
    let summary = loading.last_summary.as_str();
    let learning = ml_stats
        .map(format_loading_stats)
        .unwrap_or_else(|| format!("Using {} runtime", ml_runtime_label));
    let runtime = format!("Runtime: {}", ml_runtime_label);

    let center_y = rows / 2;
    render_centered_text(writer, center_y.saturating_sub(3), cols, &title, Color::Magenta)?;
    render_centered_text(writer, center_y.saturating_sub(1), cols, subtitle, Color::Cyan)?;
    render_centered_text(writer, center_y, cols, &progress, Color::White)?;
    render_centered_text(writer, center_y.saturating_add(2), cols, &detail, Color::White)?;
    render_centered_text(writer, center_y.saturating_add(3), cols, summary, Color::Grey)?;
    render_centered_text(writer, center_y.saturating_add(5), cols, &learning, Color::Cyan)?;
    render_centered_text(writer, rows.saturating_sub(2), cols, &runtime, Color::DarkGrey)?;

    queue!(writer, ResetColor)?;
    writer.flush()?;
    Ok(())
}

pub fn draw_boot_screen<W: Write>(
    writer: &mut W,
    diagnostics: &BootDiagnostics,
    progress_ratio: f32,
    pulse: usize,
) -> std::io::Result<()> {
    let (cols, rows) = terminal::size()?;
    queue!(writer, MoveTo(0, 0), Clear(ClearType::All))?;

    let progress = progress_bar(progress_ratio, (cols as usize).saturating_sub(22).clamp(18, 42));
    let spinner = ["/", "-", "\\", "|"][pulse % 4];
    let prompt = format!("labyrinthine-boot[1]: Running preflight diagnostics {}", spinner);
    let runtime = format!(
        "terminal={}x{}  seed={:016x}",
        diagnostics.terminal_cols,
        diagnostics.terminal_rows,
        diagnostics.session_seed
    );
    let hint = "Press any key to continue, or q to quit";

    let log_lines = vec![
        boot_result_line(
            0.081,
            "system",
            format!("terminal check passed: {} cols x {} rows", diagnostics.terminal_cols, diagnostics.terminal_rows),
            diagnostics.terminal_cols > 0 && diagnostics.terminal_rows > 0,
            pulse,
        ),
        boot_optional_line(
            0.243,
            "cpu",
            diagnostics.available_threads.zip(diagnostics.configured_threads).map(|(available, configured)| {
                format!("cpu check passed: {} threads available, worker pool set to {}", available, configured)
            }),
            pulse,
            Color::White,
        ),
        boot_probe_line(0.511, "cuda", diagnostics.cuda_available, pulse),
        boot_probe_line(0.742, "rocm", diagnostics.rocm_available, pulse),
        boot_probe_line(0.988, "vulkan", diagnostics.vulkan_available, pulse),
        boot_optional_line(
            1.286,
            "ml-runtime",
            diagnostics.runtime_detail.as_ref().zip(diagnostics.runtime_label.as_ref()).map(|(detail, label)| {
                if detail.contains("fallback") {
                    format!("ml runtime ready: {} with cpu fallback", label)
                } else if detail.contains("cpu-only") {
                    format!("ml runtime ready: {} in cpu-only mode", label)
                } else {
                    format!("ml runtime ready: {}", label)
                }
            }),
            pulse,
            if diagnostics.runtime_label.as_deref() == Some("CPU") {
                Color::Yellow
            } else {
                Color::Cyan
            },
        ),
    ];

    let start_x = 2;
    let start_y = 2;
    render_colored_text(writer, start_x, start_y, &prompt, Color::Magenta)?;
    render_colored_text(writer, start_x, start_y + 1, &runtime, Color::DarkGrey)?;

    for (idx, (line, color)) in log_lines.iter().enumerate() {
        render_colored_text(writer, start_x, start_y + 3 + idx as u16, line, *color)?;
    }

    let bar_y = (start_y + 10).min(rows.saturating_sub(3));
    render_colored_text(
        writer,
        start_x,
        bar_y,
        &format!("{}  {:>3}%", progress, (progress_ratio * 100.0).round() as u32),
        Color::White,
    )?;
    render_colored_text(writer, start_x, rows.saturating_sub(2), hint, Color::DarkGrey)?;

    queue!(writer, ResetColor)?;
    writer.flush()?;
    Ok(())
}

fn boot_stamp(seconds: f32) -> String {
    format!("[{seconds:>7.3}]", seconds = seconds)
}

fn boot_result_line(
    timestamp: f32,
    target: &str,
    message: String,
    ok: bool,
    pulse: usize,
) -> (String, Color) {
    if ok {
        (
            format!("{} {}: {}", boot_stamp(timestamp), target, message),
            Color::White,
        )
    } else {
        (
            format!("{} {}: checking{}", boot_stamp(timestamp), target, ".".repeat(pulse + 1)),
            Color::Grey,
        )
    }
}

fn boot_probe_line(
    timestamp: f32,
    target: &str,
    available: Option<bool>,
    pulse: usize,
) -> (String, Color) {
    match available {
        Some(true) => (
            format!("{} {}: detected and ready", boot_stamp(timestamp), target),
            Color::Cyan,
        ),
        Some(false) => (
            format!("{} {}: not detected", boot_stamp(timestamp), target),
            Color::DarkGrey,
        ),
        None => (
            format!("{} {}: probing{}", boot_stamp(timestamp), target, ".".repeat(pulse + 1)),
            Color::Grey,
        ),
    }
}

fn boot_optional_line(
    timestamp: f32,
    target: &str,
    message: Option<String>,
    pulse: usize,
    color: Color,
) -> (String, Color) {
    match message {
        Some(message) => (
            format!("{} {}: {}", boot_stamp(timestamp), target, message),
            color,
        ),
        None => (
            format!("{} {}: checking{}", boot_stamp(timestamp), target, ".".repeat(pulse + 1)),
            Color::Grey,
        ),
    }
}

fn hud_lines_with_level(
    game: &GameState,
    level: usize,
    _player_wins: usize,
    _agent_wins: usize,
    ml_runtime_label: &str,
    status_message: Option<&str>,
    ml_time_limit_secs: u64,
    ml_step_limit: u64,
    ml_stats: Option<MlHudStats>,
    view_mode: LevelViewMode,
    max_width: usize,
) -> (String, String) {
    let mode = match game.player.control_mode {
        ControlMode::Manual => "Manual",
        ControlMode::AutoSolve => "Auto solve",
        ControlMode::MLAgent => "ML agent",
    };
    let progress = progress_bar(game.progress_ratio(), 24);
    let progress_pct = (game.progress_ratio() * 100.0).round() as u32;

    let won_status = if game.player.won {
        if game.player.control_mode == ControlMode::MLAgent {
            "Solved. Loading the next level soon"
        } else {
            "Solved. Press N for the next level"
        }
    } else {
        "In progress"
    };
    let message = status_message
        .map(human_status_message)
        .map(|msg| format!(" | {}", msg))
        .unwrap_or_default();
    let ml_telemetry = ml_stats
        .filter(|_| game.player.control_mode == ControlMode::MLAgent)
        .map(format_ml_stats)
        .map(|stats| format!(" | {}", stats))
        .unwrap_or_default();

    let line_one = fit_status_line(
        format!(
            "Labyrinthine | Level {:02} | {} | {} | {}{}{}{}",
            level,
            mode,
            ml_runtime_label,
            won_status,
            if game.player.control_mode == ControlMode::MLAgent {
                format!(
                    " | view {}",
                    match view_mode {
                        LevelViewMode::Normal => "full",
                        LevelViewMode::AgentFocus => "agent",
                    }
                )
            } else {
                String::new()
            },
            if game.player.control_mode == ControlMode::MLAgent && !game.player.won {
                format!(" | limits {} steps / {}s", ml_step_limit, ml_time_limit_secs)
            } else {
                String::new()
            },
            ml_stats
                .filter(|_| game.player.control_mode == ControlMode::MLAgent)
                .map(|stats| format!(" | eps {:.2}", stats.epsilon))
                .unwrap_or_default(),
        ),
        max_width,
    );
    let line_two = fit_status_line(
        format!(
            "{} {:>3}% | {} steps | {}s elapsed{}{}",
            progress,
            progress_pct,
            game.player.steps,
            game.elapsed_secs(),
            ml_telemetry,
            message,
        ),
        max_width,
    );

    (line_one, line_two)
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

fn hud_primary_color(game: &GameState) -> Color {
    if game.player.won {
        Color::Green
    } else {
        match game.player.control_mode {
            ControlMode::Manual => Color::White,
            ControlMode::AutoSolve => Color::Cyan,
            ControlMode::MLAgent => Color::Magenta,
        }
    }
}

fn hud_secondary_color(game: &GameState) -> Color {
    if game.player.won {
        Color::Cyan
    } else {
        match game.player.control_mode {
            ControlMode::Manual => Color::Grey,
            ControlMode::AutoSolve => Color::Blue,
            ControlMode::MLAgent => Color::Cyan,
        }
    }
}

fn render_colored_text<W: Write>(
    writer: &mut W,
    x: u16,
    y: u16,
    text: &str,
    color: Color,
) -> std::io::Result<()> {
    queue!(writer, MoveTo(x, y), SetForegroundColor(color), Print(text), ResetColor)
}

fn render_centered_text<W: Write>(
    writer: &mut W,
    y: u16,
    total_width: u16,
    text: &str,
    color: Color,
) -> std::io::Result<()> {
    let text_width = text.chars().count().min(total_width as usize) as u16;
    let x = total_width.saturating_sub(text_width) / 2;
    render_colored_text(writer, x, y, text, color)
}

fn render_tile_row<W: Write>(writer: &mut W, x: u16, y: u16, tiles: &[Tile]) -> std::io::Result<()> {
    queue!(writer, MoveTo(x, y))?;
    let mut current_color = None;

    for tile in tiles {
        if current_color != Some(tile.color) {
            queue!(writer, SetForegroundColor(tile.color))?;
            current_color = Some(tile.color);
        }
        queue!(writer, Print(tile.glyph))?;
    }

    queue!(writer, ResetColor)
}

struct Viewport {
    offset_x: usize,
    offset_y: usize,
    origin_x: usize,
    origin_y: usize,
    visible_w: usize,
    visible_h: usize,
}

fn human_status_message(message: &str) -> String {
    if message.contains("time limit") {
        "Agent reset after reaching the time limit".to_string()
    } else if message.contains("step limit") {
        "Agent reset after reaching the step limit".to_string()
    } else if message.contains("cpu fallback") {
        "Using CPU fallback".to_string()
    } else if message.contains("cpu-only") {
        "Using CPU only".to_string()
    } else if message.contains("cuda ready") {
        "CUDA ready".to_string()
    } else if message.contains("rocm ready") {
        "ROCm ready".to_string()
    } else if message.contains("vk ready") {
        "Vulkan ready".to_string()
    } else {
        message.to_string()
    }
}

fn progress_bar(progress: f32, width: usize) -> String {
    let width = width.max(1);
    let progress = progress.clamp(0.0, 1.0);
    let filled = (progress * width as f32).floor() as usize;
    let partial = ((progress * width as f32 - filled as f32) * 4.0).round() as usize;
    let mut bar = String::with_capacity(width + 2);
    bar.push('⟦');
    for idx in 0..width {
        let glyph = if idx < filled {
            '█'
        } else if idx == filled && partial >= 2 {
            '▓'
        } else {
            '░'
        };
        bar.push(glyph);
    }
    bar.push('⟧');
    bar
}

fn format_ml_stats(stats: MlHudStats) -> String {
    let best = stats
        .best_episode_steps
        .map(|steps| steps.to_string())
        .unwrap_or_else(|| "--".to_string());

    format!(
        "learning {} runs, {} wins, {} resets, best {} steps",
        stats.episodes,
        stats.wins,
        stats.failures,
        best,
    )
}

fn format_loading_stats(stats: MlHudStats) -> String {
    let best = stats
        .best_episode_steps
        .map(|steps| format!("best solve {} steps", steps))
        .unwrap_or_else(|| "still searching for a first solve".to_string());

    format!(
        "Learning now: {} runs, {} wins, {} resets, exploration {:.2}, {}",
        stats.episodes,
        stats.wins,
        stats.failures,
        stats.epsilon,
        best,
    )
}

fn compute_viewport(
    map_w: usize,
    map_h: usize,
    view_w: usize,
    view_h: usize,
    focus: (usize, usize),

) -> Viewport {
    let visible_w = map_w.min(view_w);
    let visible_h = map_h.min(view_h);

    let mut offset_x = focus.0.saturating_sub(visible_w / 2);
    let mut offset_y = focus.1.saturating_sub(visible_h / 2);

    if offset_x + visible_w > map_w {
        offset_x = map_w.saturating_sub(visible_w);
    }
    if offset_y + visible_h > map_h {
        offset_y = map_h.saturating_sub(visible_h);
    }

    Viewport {
        offset_x,
        offset_y,
        origin_x: view_w.saturating_sub(visible_w) / 2,
        origin_y: view_h.saturating_sub(visible_h) / 2,
        visible_w,
        visible_h,
    }
}

fn build_map(maze: &Maze, game: &GameState, view_mode: LevelViewMode) -> Vec<Vec<Tile>> {
    let map_w = maze.width() * 2 + 1;
    let map_h = maze.height() * 2 + 1;
    let mut map = vec![vec![tile('█', Color::DarkBlue); map_w]; map_h];
    const TRAIL_CHARS: [char; 4] = ['.', '·', '•', 'o'];
    const TRAIL_COLORS: [Color; 4] = [Color::DarkGrey, Color::Grey, Color::Cyan, Color::White];

    for y in 0..maze.height() {
        for x in 0..maze.width() {
            let render_x = x * 2 + 1;
            let render_y = y * 2 + 1;
            map[render_y][render_x] = tile(' ', Color::Black);

            let cell = maze.cell(x, y);
            if !cell.has_wall(Direction::East) {
                map[render_y][render_x + 1] = tile(' ', Color::Black);
            }
            if !cell.has_wall(Direction::South) {
                map[render_y + 1][render_x] = tile(' ', Color::Black);
            }
        }
    }

    if game.player.control_mode == ControlMode::AutoSolve {
        for point in &game.autosolve_path {
            let x = point.0 * 2 + 1;
            let y = point.1 * 2 + 1;
            map[y][x] = tile('·', Color::Cyan);
        }
    }

    if game.player.control_mode == ControlMode::MLAgent {
        let trail_len = game.trail.len().max(1);
        for (index, point) in game.trail.iter().enumerate() {
            let x = point.0 * 2 + 1;
            let y = point.1 * 2 + 1;
            let intensity = ((index + 1) * TRAIL_CHARS.len()) / trail_len;
            let char_index = intensity.saturating_sub(1).min(TRAIL_CHARS.len() - 1);
            map[y][x] = tile(TRAIL_CHARS[char_index], TRAIL_COLORS[char_index]);
        }
    }

    let start = (maze.start.0 * 2 + 1, maze.start.1 * 2 + 1);
    let exit = (maze.exit.0 * 2 + 1, maze.exit.1 * 2 + 1);
    map[start.1][start.0] = tile('S', Color::Green);
    map[exit.1][exit.0] = tile('E', Color::Red);

    let pawn = (game.player.position.0 * 2 + 1, game.player.position.1 * 2 + 1);
    map[pawn.1][pawn.0] = tile('@', Color::Yellow);

    if view_mode == LevelViewMode::AgentFocus {
        apply_agent_focus_mask(&mut map, game);
    }

    map
}

fn apply_agent_focus_mask(map: &mut [Vec<Tile>], game: &GameState) {
    let player_render = (game.player.position.0 * 2 + 1, game.player.position.1 * 2 + 1);

    for (render_y, row) in map.iter_mut().enumerate() {
        for (render_x, tile_slot) in row.iter_mut().enumerate() {
            if !is_visible_in_agent_view((render_x, render_y), player_render) {
                *tile_slot = tile(' ', Color::Black);
            }
        }
    }

    map[player_render.1][player_render.0] = tile('@', Color::Yellow);
}

fn is_visible_in_agent_view(render: (usize, usize), player_render: (usize, usize)) -> bool {
    let render_cell = (render.0 / 2, render.1 / 2);
    let player_cell = (player_render.0 / 2, player_render.1 / 2);

    render_cell.0.abs_diff(player_cell.0) <= AGENT_FOCUS_RADIUS_X
        && render_cell.1.abs_diff(player_cell.1) <= AGENT_FOCUS_RADIUS_Y
}

fn tile(glyph: char, color: Color) -> Tile {
    Tile { glyph, color }
}

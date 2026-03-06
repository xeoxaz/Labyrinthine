use std::io::Write;

use crossterm::{
    cursor::MoveTo,
    queue,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{self, Clear, ClearType},
};

use crate::core::grid::{Direction, Maze};
use crate::play::level_game::{LevelLoadingState, ReloadState};
use crate::play::ml_solver::MlHudStats;
use crate::play::state::{ControlMode, GameState};

const AGENT_FOCUS_RADIUS: usize = 2;
const AUTHOR_CREDIT: &str = "Created by xeoxaz | github.com/xeoxaz";

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

#[derive(Clone, Copy)]
struct SeedPalette {
    wall: Color,
    primary: Color,
    accent: Color,
    auto: Color,
    muted: Color,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelViewMode {
    Normal,
    AgentFocus,
}

struct HudCorners {
    top_left: String,
    top_right: String,
    bottom_left: String,
    bottom_right: String,
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
    session_seed: u64,
) -> std::io::Result<()> {
    let (cols, rows) = terminal::size()?;
    let palette = palette_from_seed(session_seed);
    let map = build_map(&game.maze, game, view_mode, palette);
    queue!(writer, MoveTo(0, 0), Clear(ClearType::All))?;

    let hud = hud_corners_with_level(
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

    render_hud_corners(writer, cols, rows, &hud, game, palette)?;

    let view_h = rows.saturating_sub(4) as usize;
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
            (2 + viewport.origin_y + y) as u16,
            &row_tiles,
        )?;
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
    session_seed: u64,
) -> std::io::Result<()> {
    let (cols, rows) = terminal::size()?;
    queue!(writer, MoveTo(0, 0), Clear(ClearType::All))?;
    let palette = palette_from_seed(session_seed);

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
    render_centered_text(writer, center_y.saturating_sub(3), cols, &title, palette.primary)?;
    render_centered_text(writer, center_y.saturating_sub(1), cols, subtitle, palette.accent)?;
    render_centered_text(writer, center_y, cols, &progress, Color::White)?;
    render_centered_text(writer, center_y.saturating_add(2), cols, &detail, Color::White)?;
    render_centered_text(writer, center_y.saturating_add(3), cols, summary, palette.muted)?;
    render_centered_text(writer, center_y.saturating_add(5), cols, &learning, palette.accent)?;
    render_centered_text(writer, rows.saturating_sub(3), cols, AUTHOR_CREDIT, palette.muted)?;
    render_centered_text(writer, rows.saturating_sub(2), cols, &runtime, palette.muted)?;

    queue!(writer, ResetColor)?;
    writer.flush()?;
    Ok(())
}

pub fn draw_reload_screen<W: Write>(
    writer: &mut W,
    reload: &ReloadState,
    ml_runtime_label: &str,
    session_seed: u64,
) -> std::io::Result<()> {
    let (cols, rows) = terminal::size()?;
    queue!(writer, MoveTo(0, 0), Clear(ClearType::All))?;
    let palette = palette_from_seed(session_seed);

    let title = format!("Reloading Level {:02}", reload.level);
    let subtitle = "Reset triggered, rebuilding a fresh maze state";
    let progress = progress_bar(0.65, (cols as usize).saturating_sub(24).clamp(16, 36));
    let reason = compact_reload_reason(&reload.reason);
    let runtime = format!("Runtime: {}", ml_runtime_label);

    let center_y = rows / 2;
    render_centered_text(writer, center_y.saturating_sub(3), cols, &title, palette.primary)?;
    render_centered_text(writer, center_y.saturating_sub(1), cols, subtitle, palette.accent)?;
    render_centered_text(writer, center_y, cols, &progress, Color::White)?;
    render_centered_text(writer, center_y.saturating_add(2), cols, &reason, palette.muted)?;
    render_centered_text(writer, rows.saturating_sub(3), cols, AUTHOR_CREDIT, palette.muted)?;
    render_centered_text(writer, rows.saturating_sub(2), cols, &runtime, palette.muted)?;

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
    let palette = palette_from_seed(diagnostics.session_seed);

    let progress = progress_bar(progress_ratio, (cols as usize).saturating_sub(22).clamp(18, 42));
    let spinner = ["/", "-", "\\", "|"][pulse % 4];
    let prompt = format!("labyrinthine-boot[1]: Running preflight diagnostics {}", spinner);
    let runtime = format!(
        "terminal={}x{}  seed={:016x}",
        diagnostics.terminal_cols,
        diagnostics.terminal_rows,
        diagnostics.session_seed
    );
    let author = format!("author={}", AUTHOR_CREDIT);
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
                palette.accent
            },
        ),
    ];

    let start_x = 2;
    let start_y = 2;
    render_colored_text(writer, start_x, start_y, &prompt, palette.primary)?;
    render_colored_text(writer, start_x, start_y + 1, &runtime, palette.muted)?;
    render_colored_text(writer, start_x, start_y + 2, &author, palette.muted)?;

    for (idx, (line, color)) in log_lines.iter().enumerate() {
        render_colored_text(writer, start_x, start_y + 4 + idx as u16, line, *color)?;
    }

    let bar_y = (start_y + 11).min(rows.saturating_sub(3));
    render_colored_text(
        writer,
        start_x,
        bar_y,
        &format!("{}  {:>3}%", progress, (progress_ratio * 100.0).round() as u32),
        palette.accent,
    )?;
    render_colored_text(writer, start_x, rows.saturating_sub(2), hint, palette.muted)?;

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

fn hud_corners_with_level(
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
) -> HudCorners {
    let mode = match game.player.control_mode {
        ControlMode::Manual => "MAN",
        ControlMode::AutoSolve => "AUTO",
        ControlMode::MLAgent => "ML",
    };
    let segment_width = max_width.saturating_sub(6) / 2;
    let progress = progress_bar(game.progress_ratio(), 18);
    let progress_pct = (game.progress_ratio() * 100.0).round() as u32;
    let steps_bar = ratio_bar(
        if game.player.control_mode == ControlMode::MLAgent && !game.player.won {
            game.player.steps as f32 / ml_step_limit.max(1) as f32
        } else {
            0.0
        },
        8,
    );
    let time_bar = ratio_bar(
        if game.player.control_mode == ControlMode::MLAgent && !game.player.won {
            game.elapsed_secs() as f32 / ml_time_limit_secs.max(1) as f32
        } else {
            0.0
        },
        8,
    );

    let won_status = if game.player.won {
        if game.player.control_mode == ControlMode::MLAgent {
            "WIN>AUTO"
        } else {
            "WIN>N"
        }
    } else {
        "LIVE"
    };
    let message = status_message.map(compact_status_message).unwrap_or_default();
    let ml_telemetry = ml_stats
        .filter(|_| game.player.control_mode == ControlMode::MLAgent)
        .map(format_ml_stats)
        .unwrap_or_default();

    let top_left = fit_status_line(
        format!(
            "LAB | L{:02} | {} | {} | {}",
            level,
            mode,
            ml_runtime_label,
            won_status,
        ),
        segment_width,
    );

    let top_right = fit_status_line(
        format!(
            "{}{}{}",
            if game.player.control_mode == ControlMode::MLAgent {
                format!(
                    "V:{}",
                    match view_mode {
                        LevelViewMode::Normal => "MAP",
                        LevelViewMode::AgentFocus => "NEAR",
                    }
                )
            } else {
                String::new()
            },
            if game.player.control_mode == ControlMode::MLAgent && !game.player.won {
                format!(
                    "{}S{} T{}",
                    if game.player.control_mode == ControlMode::MLAgent {
                        " | "
                    } else {
                        ""
                    },
                    steps_bar,
                    time_bar
                )
            } else {
                String::new()
            },
            ml_stats
                .filter(|_| game.player.control_mode == ControlMode::MLAgent)
                .map(|stats| format!(" | E{}", ratio_bar(1.0 - stats.epsilon, 6)))
                .unwrap_or_default(),
        ),
        segment_width,
    );

    let bottom_left = fit_status_line(
        format!(
            "P{} {:>3}% | {:>4} | {:>3}s",
            progress,
            progress_pct,
            game.player.steps,
            game.elapsed_secs(),
        ),
        segment_width,
    );

    let bottom_right = fit_status_line(
        match (ml_telemetry.is_empty(), message.is_empty()) {
            (false, false) => format!("{} | {}", ml_telemetry, message),
            (false, true) => ml_telemetry,
            (true, false) => message,
            (true, true) => String::new(),
        },
        segment_width,
    );

    HudCorners {
        top_left,
        top_right,
        bottom_left,
        bottom_right,
    }
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

fn hud_primary_color(game: &GameState, palette: SeedPalette) -> Color {
    if game.player.won {
        Color::Green
    } else {
        match game.player.control_mode {
            ControlMode::Manual => Color::White,
            ControlMode::AutoSolve => palette.auto,
            ControlMode::MLAgent => palette.primary,
        }
    }
}

fn hud_secondary_color(game: &GameState, palette: SeedPalette) -> Color {
    if game.player.won {
        palette.accent
    } else {
        match game.player.control_mode {
            ControlMode::Manual => palette.muted,
            ControlMode::AutoSolve => palette.auto,
            ControlMode::MLAgent => palette.accent,
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

fn render_right_aligned_text<W: Write>(
    writer: &mut W,
    y: u16,
    total_width: u16,
    text: &str,
    color: Color,
    padding: u16,
) -> std::io::Result<()> {
    let text_width = text.chars().count().min(total_width as usize) as u16;
    let x = total_width.saturating_sub(text_width.saturating_add(padding));
    render_colored_text(writer, x, y, text, color)
}

fn render_hud_corners<W: Write>(
    writer: &mut W,
    cols: u16,
    rows: u16,
    hud: &HudCorners,
    game: &GameState,
    palette: SeedPalette,
) -> std::io::Result<()> {
    let primary = hud_primary_color(game, palette);
    let secondary = hud_secondary_color(game, palette);

    render_colored_text(writer, 1, 0, &hud.top_left, primary)?;
    render_right_aligned_text(writer, 0, cols, &hud.top_right, primary, 1)?;

    if rows > 1 {
        let bottom_y = rows - 1;
        render_colored_text(writer, 1, bottom_y, &hud.bottom_left, secondary)?;
        render_right_aligned_text(writer, bottom_y, cols, &hud.bottom_right, secondary, 1)?;
    }

    Ok(())
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

fn compact_status_message(message: &str) -> String {
    if message.contains("time limit") {
        "RST:T".to_string()
    } else if message.contains("step limit") {
        "RST:S".to_string()
    } else if message.contains("cpu fallback") {
        "CPU:FALLBACK".to_string()
    } else if message.contains("cpu-only") {
        "CPU".to_string()
    } else if message.contains("cuda ready") {
        "CUDA".to_string()
    } else if message.contains("rocm ready") {
        "ROCM".to_string()
    } else if message.contains("vk ready") {
        "VK".to_string()
    } else if message.contains("loading level") {
        "LOAD".to_string()
    } else if message.contains("level") && message.contains("ready") {
        "READY".to_string()
    } else {
        message.to_uppercase()
    }
}

fn compact_reload_reason(message: &str) -> String {
    if message.contains("time limit") {
        "Reload reason: time limit reached".to_string()
    } else if message.contains("step limit") {
        "Reload reason: step limit reached".to_string()
    } else {
        format!("Reload reason: {}", message)
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

fn ratio_bar(ratio: f32, width: usize) -> String {
    progress_bar(ratio.clamp(0.0, 1.0), width)
}

fn format_ml_stats(stats: MlHudStats) -> String {
    format!(
        "R{} W{} F{} B{}",
        stats.episodes,
        stats.wins,
        stats.failures,
        stats
            .best_episode_steps
            .map(|steps| steps.to_string())
            .unwrap_or_else(|| "--".to_string()),
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

fn build_map(
    maze: &Maze,
    game: &GameState,
    view_mode: LevelViewMode,
    palette: SeedPalette,
) -> Vec<Vec<Tile>> {
    let map_w = maze.width() * 2 + 1;
    let map_h = maze.height() * 2 + 1;
    let mut map = vec![vec![tile('█', palette.wall); map_w]; map_h];

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
            map[y][x] = tile('·', palette.accent);
        }
    }

    let start = (maze.start.0 * 2 + 1, maze.start.1 * 2 + 1);
    let exit = (maze.exit.0 * 2 + 1, maze.exit.1 * 2 + 1);
    map[start.1][start.0] = tile('S', Color::DarkGreen);
    map[exit.1][exit.0] = tile('E', Color::DarkRed);

    let pawn = (game.player.position.0 * 2 + 1, game.player.position.1 * 2 + 1);
    let (pawn_glyph, pawn_color) = if game.player.control_mode == ControlMode::MLAgent {
        ('◉', Color::White)
    } else {
        ('@', Color::Yellow)
    };
    map[pawn.1][pawn.0] = tile(pawn_glyph, pawn_color);

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

    map[player_render.1][player_render.0] = tile('◉', Color::White);
}

fn is_visible_in_agent_view(render: (usize, usize), player_render: (usize, usize)) -> bool {
    let render_cell = (render.0 / 2, render.1 / 2);
    let player_cell = (player_render.0 / 2, player_render.1 / 2);

    let dx = render_cell.0.abs_diff(player_cell.0) as u32;
    let dy = render_cell.1.abs_diff(player_cell.1) as u32;
    let radius = AGENT_FOCUS_RADIUS as u32;

    dx * dx + dy * dy <= radius * radius
}

fn tile(glyph: char, color: Color) -> Tile {
    Tile { glyph, color }
}

fn palette_from_seed(seed: u64) -> SeedPalette {
    const PALETTES: [SeedPalette; 5] = [
        SeedPalette {
            wall: Color::DarkBlue,
            primary: Color::Blue,
            accent: Color::Grey,
            auto: Color::Blue,
            muted: Color::DarkGrey,
        },
        SeedPalette {
            wall: Color::DarkGreen,
            primary: Color::DarkGreen,
            accent: Color::Grey,
            auto: Color::Green,
            muted: Color::DarkGrey,
        },
        SeedPalette {
            wall: Color::DarkRed,
            primary: Color::DarkRed,
            accent: Color::Grey,
            auto: Color::DarkMagenta,
            muted: Color::DarkGrey,
        },
        SeedPalette {
            wall: Color::DarkCyan,
            primary: Color::DarkCyan,
            accent: Color::Grey,
            auto: Color::Cyan,
            muted: Color::DarkGrey,
        },
        SeedPalette {
            wall: Color::DarkMagenta,
            primary: Color::DarkMagenta,
            accent: Color::Grey,
            auto: Color::DarkCyan,
            muted: Color::DarkGrey,
        },
    ];

    PALETTES[(seed as usize) % PALETTES.len()]
}

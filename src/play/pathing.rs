use crate::core::{grid::Maze, solver};

pub fn build_autosolve_path(maze: &Maze) -> Vec<(usize, usize)> {
    solver::shortest_path(maze, maze.start, maze.exit).unwrap_or_else(|| vec![maze.start])
}

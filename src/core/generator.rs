use rand::prelude::IndexedRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;

use super::grid::{Direction, Maze};

pub fn generate_recursive_backtracker(width: usize, height: usize, seed: u64) -> Maze {
    let mut maze = Maze::new(width.max(2), height.max(2));
    let mut visited = vec![false; maze.width() * maze.height()];
    let mut stack = Vec::with_capacity(maze.width() * maze.height());
    let mut rng = StdRng::seed_from_u64(seed);

    let start = maze.start;
    visited[maze.index(start.0, start.1)] = true;
    stack.push(start);

    while let Some(&(x, y)) = stack.last() {
        let mut candidates: Vec<(Direction, (usize, usize))> = Vec::new();
        for direction in Direction::ALL {
            if let Some((nx, ny)) = maze.neighbor(x, y, direction) {
                let idx = maze.index(nx, ny);
                if !visited[idx] {
                    candidates.push((direction, (nx, ny)));
                }
            }
        }

        if let Some(&(direction, next)) = candidates.choose(&mut rng) {
            maze.remove_wall_between((x, y), next, direction);
            visited[maze.index(next.0, next.1)] = true;
            stack.push(next);
        } else {
            stack.pop();
        }
    }

    maze
}

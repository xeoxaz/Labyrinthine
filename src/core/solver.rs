use std::collections::VecDeque;

use super::grid::{Direction, Maze};

pub fn shortest_path_distance(
    maze: &Maze,
    start: (usize, usize),
    goal: (usize, usize),
) -> Option<usize> {
    if start == goal {
        return Some(0);
    }

    let total = maze.width() * maze.height();
    let mut visited = vec![false; total];
    let mut queue = VecDeque::new();

    visited[maze.index(start.0, start.1)] = true;
    queue.push_back((start, 0usize));

    while let Some((current, distance)) = queue.pop_front() {
        for direction in Direction::ALL {
            if let Some(next) = maze.can_move(current, direction) {
                if next == goal {
                    return Some(distance + 1);
                }

                let idx = maze.index(next.0, next.1);
                if visited[idx] {
                    continue;
                }

                visited[idx] = true;
                queue.push_back((next, distance + 1));
            }
        }
    }

    None
}

pub fn shortest_path(
    maze: &Maze,
    start: (usize, usize),
    goal: (usize, usize),
) -> Option<Vec<(usize, usize)>> {
    let total = maze.width() * maze.height();
    let mut visited = vec![false; total];
    let mut parent: Vec<Option<(usize, usize)>> = vec![None; total];
    let mut queue = VecDeque::new();

    visited[maze.index(start.0, start.1)] = true;
    queue.push_back(start);

    while let Some(current) = queue.pop_front() {
        if current == goal {
            return Some(reconstruct_path(maze, parent, start, goal));
        }

        for direction in Direction::ALL {
            if let Some(next) = maze.can_move(current, direction) {
                let idx = maze.index(next.0, next.1);
                if visited[idx] {
                    continue;
                }

                visited[idx] = true;
                parent[idx] = Some(current);
                queue.push_back(next);
            }
        }
    }

    None
}

fn reconstruct_path(
    maze: &Maze,
    parent: Vec<Option<(usize, usize)>>,
    start: (usize, usize),
    goal: (usize, usize),
) -> Vec<(usize, usize)> {
    let mut path = vec![goal];
    let mut current = goal;

    while current != start {
        let index = maze.index(current.0, current.1);
        if let Some(prev) = parent[index] {
            current = prev;
            path.push(current);
        } else {
            break;
        }
    }

    path.reverse();
    path
}

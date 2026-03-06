#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    North,
    East,
    South,
    West,
}

impl Direction {
    pub const ALL: [Direction; 4] = [
        Direction::North,
        Direction::East,
        Direction::South,
        Direction::West,
    ];

    pub fn opposite(self) -> Direction {
        match self {
            Direction::North => Direction::South,
            Direction::East => Direction::West,
            Direction::South => Direction::North,
            Direction::West => Direction::East,
        }
    }

    pub fn delta(self) -> (isize, isize) {
        match self {
            Direction::North => (0, -1),
            Direction::East => (1, 0),
            Direction::South => (0, 1),
            Direction::West => (-1, 0),
        }
    }

    pub fn index(self) -> usize {
        match self {
            Direction::North => 0,
            Direction::East => 1,
            Direction::South => 2,
            Direction::West => 3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Cell {
    walls: [bool; 4],
}

impl Cell {
    fn new() -> Self {
        Self { walls: [true; 4] }
    }

    pub fn has_wall(&self, direction: Direction) -> bool {
        self.walls[direction.index()]
    }

    fn set_wall(&mut self, direction: Direction, value: bool) {
        self.walls[direction.index()] = value;
    }
}

#[derive(Clone, Debug)]
pub struct Maze {
    width: usize,
    height: usize,
    cells: Vec<Cell>,
    pub start: (usize, usize),
    pub exit: (usize, usize),
}

impl Maze {
    pub fn new(width: usize, height: usize) -> Self {
        let size = width.saturating_mul(height);
        let cells = vec![Cell::new(); size];
        let start = (0, 0);
        let exit = (width.saturating_sub(1), height.saturating_sub(1));

        Self {
            width,
            height,
            cells,
            start,
            exit,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn cell(&self, x: usize, y: usize) -> &Cell {
        &self.cells[self.index(x, y)]
    }

    fn cell_mut(&mut self, x: usize, y: usize) -> &mut Cell {
        let index = self.index(x, y);
        &mut self.cells[index]
    }

    pub fn index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    pub fn in_bounds(&self, x: isize, y: isize) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height
    }

    pub fn neighbor(
        &self,
        x: usize,
        y: usize,
        direction: Direction,
    ) -> Option<(usize, usize)> {
        let (dx, dy) = direction.delta();
        let nx = x as isize + dx;
        let ny = y as isize + dy;

        if self.in_bounds(nx, ny) {
            Some((nx as usize, ny as usize))
        } else {
            None
        }
    }

    pub fn remove_wall_between(
        &mut self,
        first: (usize, usize),
        second: (usize, usize),
        direction: Direction,
    ) {
        let first_cell = self.cell_mut(first.0, first.1);
        first_cell.set_wall(direction, false);

        let second_cell = self.cell_mut(second.0, second.1);
        second_cell.set_wall(direction.opposite(), false);
    }

    pub fn can_move(&self, from: (usize, usize), direction: Direction) -> Option<(usize, usize)> {
        if self.cell(from.0, from.1).has_wall(direction) {
            return None;
        }

        self.neighbor(from.0, from.1, direction)
    }
}

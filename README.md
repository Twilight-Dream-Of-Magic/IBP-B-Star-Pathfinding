# IBP-B-Star Pathfinding

An open-source implementation of the original  An Intelligent Bi-Directional Parallel B\* Routing Algorithm, reconstructed from the authoritative paper and provided in both Python and C++.

## Key Features

- Multi-language support:

  - Python script implementations (`IBP-B-Star Pathfinding.py`, `AStar-BStar New Implementation.py`)
  - C++ implementation with CMake (`IBP-B-Star Pathfinding.cpp`, `IBP-B-Star Pathfinding.hpp`, `main.cpp`)
- Original paper included for reference (`An Intelligent Bi-Directional Parallel B-Star Routing Algorithm.pdf`)
- Dual-mode operation: simple proof‑of‑concept vs. engineering‑grade maze support
- Licensed under GNU GPL‑3.0 for free use, modification, and distribution

## Repository Contents

- `An Intelligent Bi-Directional Parallel B-Star Routing Algorithm.pdf`
  Contains the full paper with algorithm description, pseudocode, and performance analysis.
- `IBP-B-Star Pathfinding.hpp`
  Header file defining core classes and functions for the C++ implementation.
- `IBP-B-Star Pathfinding (OldVersion).cpp`
  Early C++ prototype with basic interval evaluation logic.
- `main.cpp`
  Example driver to compile, run, and benchmark the C++ implementation.
- `CMakeLists.txt`
  Build configuration for generating the C++ executable.
- `IBP-B-Star Pathfinding.py`
  Full-featured Python implementation handling complex mazes.
- `AStar-BStar New Implementation.py`
  Lean Python version illustrating the core B\* idea alongside A\* for comparison.
- `.clang-format`
  Code style settings for consistent C/C++ formatting.
- `LICENSE`
  Text of the GNU GPL‑3.0 license governing this project.

## Performance and Comparison

- The original B\- idea excels in interval‑based expansion but may stall in complex mazes.
- The engineering‑grade version handles labyrinths reliably but runs roughly twice as slow as a tuned A\- on standard grids.
- See the paper’s benchmarks or run your own tests via `main.cpp` for detailed comparisons.

## Contributing

Contributions, bug reports, and improvements are welcome! Please fork the repository, create a feature branch, and submit a pull request.

## License

This project is released under the GNU General Public License v3.0. See `LICENSE` for full terms.

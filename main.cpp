#include "IBP-B-Star Pathfinding.hpp"

// =========================== main ===========================

int main( int argc, char** argv )
{
	std::ios::sync_with_stdio( false );
	std::cin.tie( nullptr );

	using namespace IBP_BStarAlogithm;

	if ( argc > 1 )
	{
		try
		{
			ParseArgs( argc, argv );
		}
		catch ( const std::exception& e )
		{
			std::cerr << "Error parsing args: " << e.what() << "\n";
			return 1;
		}
	}

	int attempt_counter = 0;
	while ( true )
	{
		++attempt_counter;
		Grid		 grid;
		CellPosition start_pos, goal_pos;
		try
		{
			if ( g_config.use_random_map && g_config.map_file_path.empty() )
			{
				grid = GenerateRandomGrid( g_config.random_map_width, g_config.random_map_height, g_config.random_wall_probability, g_config.random_seed + attempt_counter - 1, start_pos, goal_pos );
			}
			else
			{
				std::tie( grid, start_pos, goal_pos ) = LoadGridFromFile( g_config.map_file_path );
				if ( g_config.cli_start_override )
					start_pos = *g_config.cli_start_override;
				if ( g_config.cli_goal_override )
					goal_pos = *g_config.cli_goal_override;
			}
		}
		catch ( const std::exception& e )
		{
			std::cerr << "Map load error: " << e.what() << "\n";
			return 1;
		}

		SearchOutcome outcome = RunIbpBStar( grid, start_pos, goal_pos, g_config.flush_wait_layers );

		if ( !outcome.success || !ValidatePathContiguity( grid, outcome.final_path ) )
		{
			if ( g_config.reroll_until_solvable && g_config.use_random_map && attempt_counter < g_config.reroll_max_attempts )
			{
				continue;  // 再随机一张
			}
			if ( !outcome.success )
			{
				std::cout << "No path is reachable for the current obstacle!\n";
			}
			else
			{
				std::cout << "Invalid path reconstructed (broken chain).\n";
				if ( g_config.print_invalid_path && g_config.print_path )
				{
					RenderGridWithPath( grid, outcome.final_path, start_pos, goal_pos );
				}
			}
		}

		if ( g_config.print_stats )
		{
			std::cout << "path_length=" << outcome.statistics.final_path_length << " meet=(" << outcome.meet_position.row << ", " << outcome.meet_position.col << ")"
					  << " expanded=" << outcome.statistics.expanded_node_count << " tries=" << attempt_counter << "\n";
		}
		if ( g_config.print_path )
		{
			RenderGridWithPath( grid, outcome.final_path, start_pos, goal_pos );
		}
		MazeDemo();
		return 0;
	}
}

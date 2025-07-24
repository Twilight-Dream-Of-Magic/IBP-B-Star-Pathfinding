/*
 * IBP-B* Pathfinding (Paper-accurate) - C++17
 * An Intelligent Bi-Directional Parallel B-Star Routing Algorithm
 * https://www.scirp.org/pdf/jcc_2020062915444066.pdf
 * ------------------------------------------------------------
 * EN SUMMARY:
 *   A bi-directional, greedy-first pathfinding with special rules:
 *   - Greedy one-step move toward goal
 *   - First / multi obstacle handling
 *   - Obstacle "rebirth" (once per node when both sides of B are blocked)
 *   - Constant-time concave pre-exploration
 *   - Forward + Backward simultaneous search
 *   - Peer waiting flush levels (±WAIT_LAYERS around meet depth)
 *
 * CN 摘要：
 *   一种双向并行的贪心优先寻路算法，带有特殊规则：
 *   - 贪心方向先走一步
 *   - 第一次/多次碰壁时的不同扩展策略
 *   - “再生”机制：当 B 的左右都为障碍且尚未再生过时，当前节点重新入队一次
 *   - O(1) 凹形预探索（检测贪心方向垂直两个格）
 *   - 前向 + 后向并行搜索
 *   - 同层等待：在相遇层上下 ±WAIT_LAYERS 冲刷
 *
 * Compile:
 *   g++ -std=c++17 -O2 ibp_bstar.cpp -o ibp_bstar
 * Run:
 *   ./ibp_bstar --random 64 64 0.25 --seed 56464641 --wait 2
 *   ./ibp_bstar --map map.txt --sx 0 --sy 0 --ex 63 --ey 63
 */

#include <array>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace IBP_BStarAlogithm
{

	// =========================== 基础类型 ===========================

	struct CellPosition
	{
		int row;
		int col;
	};

	using Grid = std::vector<std::vector<int>>;

	struct RunConfig
	{
		// 地图来源配置
		bool						use_random_map = true;
		int							random_map_width = 96;
		int							random_map_height = 96;
		double						random_wall_probability = 0.18;
		std::uint32_t				random_seed = 41548941u;
		std::string					map_file_path = "";
		std::optional<CellPosition> cli_start_override;
		std::optional<CellPosition> cli_goal_override;
		bool						reroll_until_solvable = true;
		int							reroll_max_attempts = 100;

		// 算法参数
		int	 flush_wait_layers = 2;	 // 相遇层上下继续冲刷的层数
		bool print_path = true;
		bool print_stats = true;
		bool print_invalid_path = false;
		bool print_arrows = false;

		// 输出字符集
		bool use_ascii_glyphs = true;

		std::string							  WALL_U = "■";
		std::string							  EMPTY_U = "○";
		std::string							  PATH_U = "★";
		std::string							  START_U = "×";
		std::string							  END_U = "√";
		std::unordered_map<char, std::string> ARROWS_U = { { 'U', "↑" }, { 'D', "↓" }, { 'L', "←" }, { 'R', "→" } };

		std::string							  WALL_A = "+";
		std::string							  EMPTY_A = ".";
		std::string							  PATH_A = "?";
		std::string							  START_A = "S";
		std::string							  END_A = "E";
		std::unordered_map<char, std::string> ARROWS_A = { { 'U', "^" }, { 'D', "v" }, { 'L', "<" }, { 'R', ">" } };
	};

	inline RunConfig g_config;	// 全局配置对象

	// =========================== 轻量 RNG ===========================

	struct LinearCongruentialRng32
	{
		std::uint32_t state;
		explicit LinearCongruentialRng32( std::uint32_t seed ) : state( seed ) {}

		inline std::uint32_t NextUint32()
		{
			state = ( 1664525u * state + 1013904223u );
			return state;
		}
		inline double NextFloatZeroToOne()
		{
			return static_cast<double>( NextUint32() ) / 4294967296.0;
		}
	};

	// =========================== 工具常量/函数 ===========================

	using DirectionDeltaMap = std::unordered_map<char, std::pair<int, int>>;
	inline const DirectionDeltaMap kDirectionDeltaMap = { { 'U', { -1, 0 } }, { 'D', { 1, 0 } }, { 'L', { 0, -1 } }, { 'R', { 0, 1 } } };

	inline const std::array<char, 4> kDirectionPriorityOrder = { 'U', 'D', 'L', 'R' };

	inline int ToLinearIndex( int row, int col, int grid_width )
	{
		return row * grid_width + col;
	}

	inline std::pair<int, int> FromLinearIndex( int linear_index, int grid_width )
	{
		return { linear_index / grid_width, linear_index % grid_width };
	}

	inline bool IsCellPassable( const Grid& grid, int row, int col )
	{
		return row >= 0 && row < static_cast<int>( grid.size() ) && col >= 0 && col < static_cast<int>( grid[ 0 ].size() ) && grid[ row ][ col ] == 0;
	}

	inline bool IsCellBlocked( const Grid& grid, int row, int col )
	{
		return !IsCellPassable( grid, row, col );
	}

	inline char OppositeDirection( char direction_char )
	{
		switch ( direction_char )
		{
		case 'U':
			return 'D';
		case 'D':
			return 'U';
		case 'L':
			return 'R';
		default:
			return 'L';	 // direction_char == 'R'
		}
	}

	inline std::pair<char, char> LeftRightDirections( char direction_char )
	{
		// 对应 Python dict: {"U":("L","R"), "D":("R","L"), "L":("D","U"), "R":("U","D")}
		switch ( direction_char )
		{
		case 'U':
			return { 'L', 'R' };
		case 'D':
			return { 'R', 'L' };
		case 'L':
			return { 'D', 'U' };
		default:
			return { 'U', 'D' };  // direction_char == 'R'
		}
	}

	inline char ChooseGreedyDirection( int current_row, int current_col, int goal_row, int goal_col )
	{
		int delta_row = goal_row - current_row;
		int delta_col = goal_col - current_col;
		if ( std::abs( delta_row ) >= std::abs( delta_col ) )
		{
			return ( delta_row > 0 ) ? 'D' : 'U';
		}
		else
		{
			return ( delta_col > 0 ) ? 'R' : 'L';
		}
	}

	// =========================== 地图生成 / 读取 ===========================

	inline Grid GenerateRandomGrid( int width, int height, double wall_probability, std::uint32_t seed, CellPosition& out_start, CellPosition& out_goal )
	{
		LinearCongruentialRng32 rng( seed );
		Grid					grid( height, std::vector<int>( width, 0 ) );
		for ( int row_index = 0; row_index < height; ++row_index )
		{
			for ( int col_index = 0; col_index < width; ++col_index )
			{
				if ( rng.NextFloatZeroToOne() < wall_probability )
					grid[ row_index ][ col_index ] = 1;
			}
		}
		out_start = { 0, 0 };
		out_goal = { height - 1, width - 1 };
		grid[ out_start.row ][ out_start.col ] = 0;
		grid[ out_goal.row ][ out_goal.col ] = 0;
		return grid;
	}

	inline std::tuple<Grid, CellPosition, CellPosition> LoadGridFromFile( const std::string& file_path )
	{
		std::ifstream file_stream( file_path );
		if ( !file_stream )
			throw std::runtime_error( "Failed to open map file: " + file_path );

		std::vector<std::string> wall_tokens = { g_config.WALL_U, g_config.WALL_A };
		std::vector<std::string> empty_tokens = { g_config.EMPTY_U, g_config.EMPTY_A };
		std::vector<std::string> start_tokens = { g_config.START_U, g_config.START_A, "S", "s" };
		std::vector<std::string> goal_tokens = { g_config.END_U, g_config.END_A, "E", "e" };

		auto IsOneOf = []( const std::string& seg, const std::vector<std::string>& pool ) -> bool {
			for ( const auto& tok : pool )
				if ( seg == tok )
					return true;
			return false;
		};

		std::size_t max_token_length = 1;
		auto		UpdateMax = [ & ]( const std::vector<std::string>& vec ) {
			   for ( const auto& s : vec )
				   max_token_length = std::max( max_token_length, s.size() );
		};
		UpdateMax( wall_tokens );
		UpdateMax( empty_tokens );
		UpdateMax( start_tokens );
		UpdateMax( goal_tokens );

		Grid		 grid;
		CellPosition start_pos { -1, -1 }, goal_pos { -1, -1 };

		std::string raw_line;
		int			current_row_index = 0;	// for error info
		while ( std::getline( file_stream, raw_line ) )
		{
			// 去空白
			raw_line.erase( std::remove_if( raw_line.begin(), raw_line.end(), []( unsigned char ch ) { return std::isspace( ch ); } ), raw_line.end() );
			if ( raw_line.empty() )
				continue;

			std::vector<int> row_vector;
			for ( std::size_t char_pos = 0; char_pos < raw_line.size(); )
			{
				bool		token_matched = false;
				std::size_t taken_chars = 0;
				std::string token_candidate;
				std::size_t try_limit = std::min<std::size_t>( max_token_length, raw_line.size() - char_pos );

				for ( std::size_t token_len = try_limit; token_len >= 1; --token_len )
				{
					token_candidate = raw_line.substr( char_pos, token_len );
					if ( IsOneOf( token_candidate, wall_tokens ) || IsOneOf( token_candidate, empty_tokens ) || IsOneOf( token_candidate, start_tokens ) || IsOneOf( token_candidate, goal_tokens ) )
					{
						taken_chars = token_len;
						token_matched = true;
						break;
					}
				}
				if ( !token_matched )
				{
					throw std::runtime_error( "Bad glyph at row " + std::to_string( current_row_index ) + ", pos " + std::to_string( char_pos ) );
				}

				if ( IsOneOf( token_candidate, wall_tokens ) )
				{
					row_vector.push_back( 1 );
				}
				else if ( IsOneOf( token_candidate, empty_tokens ) )
				{
					row_vector.push_back( 0 );
				}
				else if ( IsOneOf( token_candidate, start_tokens ) )
				{
					start_pos = { current_row_index, static_cast<int>( row_vector.size() ) };
					row_vector.push_back( 0 );
				}
				else if ( IsOneOf( token_candidate, goal_tokens ) )
				{
					goal_pos = { current_row_index, static_cast<int>( row_vector.size() ) };
					row_vector.push_back( 0 );
				}
				char_pos += taken_chars;
			}
			grid.push_back( std::move( row_vector ) );
			++current_row_index;
		}

		if ( start_pos.row == -1 || goal_pos.row == -1 )
		{
			throw std::runtime_error( "Map needs S/E" );
		}
		return { grid, start_pos, goal_pos };
	}

	// =========================== IBP-B* 核心 ===========================

	struct SearchStatistics
	{
		int expanded_node_count = 0;
		int final_path_length = 0;
	};

	struct SearchOutcome
	{
		std::vector<CellPosition> final_path;
		CellPosition			  meet_position { -1, -1 };
		SearchStatistics		  statistics;
		bool					  success = false;
	};

	inline bool IsConcaveEntry( const Grid& grid, int base_row, int base_col, char greedy_direction )
	{
		if ( greedy_direction == 'L' || greedy_direction == 'R' )
		{
			bool cell_up_free = IsCellPassable( grid, base_row - 1, base_col );
			bool cell_down_free = IsCellPassable( grid, base_row + 1, base_col );
			return cell_up_free && cell_down_free;
		}
		else
		{
			bool cell_left_free = IsCellPassable( grid, base_row, base_col - 1 );
			bool cell_right_free = IsCellPassable( grid, base_row, base_col + 1 );
			return cell_left_free && cell_right_free;
		}
	}

	inline SearchOutcome RunIbpBStar( const Grid& grid, CellPosition start_pos, CellPosition goal_pos, int wait_layers )
	{
		const int grid_height = static_cast<int>( grid.size() );
		const int grid_width = static_cast<int>( grid[ 0 ].size() );
		const int total_cells = grid_height * grid_width;

		auto ToIndex = [ & ]( int row, int col ) -> int {
			return ToLinearIndex( row, col, grid_width );
		};
		auto FromIndex = [ & ]( int linear_index ) -> CellPosition {
			auto [ r, c ] = FromLinearIndex( linear_index, grid_width );
			return CellPosition { r, c };
		};

		using DepthArray = std::vector<int>;
		using ParentArray = std::vector<int>;
		using HitArray = std::vector<int>;
		using FlagArray = std::vector<bool>;
		using NodeQueue = std::deque<int>;

		DepthArray	depth_from_start( total_cells, -1 );
		DepthArray	depth_from_goal( total_cells, -1 );
		ParentArray parent_from_start( total_cells, -1 );
		ParentArray parent_from_goal( total_cells, -1 );
		HitArray	obstacle_hits_from_start( total_cells, 0 );
		HitArray	obstacle_hits_from_goal( total_cells, 0 );
		FlagArray	rebirth_used_from_start( total_cells, false );
		FlagArray	rebirth_used_from_goal( total_cells, false );

		NodeQueue frontier_queue_from_start;
		NodeQueue frontier_queue_from_goal;

		int start_linear_index = ToIndex( start_pos.row, start_pos.col );
		int goal_linear_index = ToIndex( goal_pos.row, goal_pos.col );

		depth_from_start[ start_linear_index ] = 0;
		depth_from_goal[ goal_linear_index ] = 0;
		frontier_queue_from_start.push_back( start_linear_index );
		frontier_queue_from_goal.push_back( goal_linear_index );

		int			  meet_linear_index = -1;
		int			  meet_depth_sum = -1;
		std::set<int> flush_depth_levels;  // 保存需要继续扩展的层深度

		SearchStatistics stats;

		// 检查是否相遇，如果是第一次相遇，则计算 flush 区间
		auto CheckMeet = [ & ]( int node_linear_index, DepthArray& this_side_depth, DepthArray& other_side_depth ) {
			if ( other_side_depth[ node_linear_index ] != -1 && meet_linear_index == -1 )
			{
				meet_linear_index = node_linear_index;
				meet_depth_sum = this_side_depth[ node_linear_index ] + other_side_depth[ node_linear_index ];

				flush_depth_levels.clear();
				flush_depth_levels.insert( meet_depth_sum );
				for ( int k_layer = 1; k_layer <= wait_layers; ++k_layer )
				{
					flush_depth_levels.insert( meet_depth_sum + k_layer );
					flush_depth_levels.insert( meet_depth_sum - k_layer );
				}
			}
		};

		// 扩展一侧前沿
		auto ExpandFrontier = [ & ]( int current_linear_index, int greedy_goal_row, int greedy_goal_col, DepthArray& this_depth, DepthArray& opposite_depth, ParentArray& this_parent, HitArray& this_obstacle_hits, NodeQueue& this_queue, FlagArray& this_rebirth_used, char /*debug_tag*/ ) {
			stats.expanded_node_count++;

			CellPosition current_pos = FromIndex( current_linear_index );
			int			 current_row = current_pos.row;
			int			 current_col = current_pos.col;

			char greedy_direction = ChooseGreedyDirection( current_row, current_col, greedy_goal_row, greedy_goal_col );
			auto direction_offset = kDirectionDeltaMap.at( greedy_direction );
			int	 step_row_offset = direction_offset.first;
			int	 step_col_offset = direction_offset.second;

			// B cell：贪心方向正前方的那个格子
			int greedy_cell_row = current_row + step_row_offset;
			int greedy_cell_col = current_col + step_col_offset;

			int current_depth_value = this_depth[ current_linear_index ];

			// 1) 贪心一步
			if ( IsCellPassable( grid, greedy_cell_row, greedy_cell_col ) )
			{
				int greedy_cell_index = ToIndex( greedy_cell_row, greedy_cell_col );
				if ( this_depth[ greedy_cell_index ] == -1 )
				{
					this_depth[ greedy_cell_index ] = current_depth_value + 1;
					this_parent[ greedy_cell_index ] = current_linear_index;
					this_obstacle_hits[ greedy_cell_index ] = this_obstacle_hits[ current_linear_index ];
					this_queue.push_back( greedy_cell_index );
					CheckMeet( greedy_cell_index, this_depth, opposite_depth );

					// 2) 常数时间凹形预探索
					if ( IsConcaveEntry( grid, greedy_cell_row, greedy_cell_col, greedy_direction ) )
					{
						for ( char scan_dir : kDirectionPriorityOrder )
						{
							if ( scan_dir == greedy_direction )
								continue;
							auto scan_offset = kDirectionDeltaMap.at( scan_dir );
							int	 neighbor_row = current_row + scan_offset.first;
							int	 neighbor_col = current_col + scan_offset.second;
							if ( IsCellPassable( grid, neighbor_row, neighbor_col ) )
							{
								int neighbor_index = ToIndex( neighbor_row, neighbor_col );
								if ( this_depth[ neighbor_index ] == -1 )
								{
									this_depth[ neighbor_index ] = current_depth_value + 1;
									this_parent[ neighbor_index ] = current_linear_index;
									this_obstacle_hits[ neighbor_index ] = this_obstacle_hits[ current_linear_index ];
									this_queue.push_back( neighbor_index );
									CheckMeet( neighbor_index, this_depth, opposite_depth );
								}
							}
						}
					}
				}
				return;	 // 贪心成功，结束本次扩展
			}

			// 2) 碰壁后再生判定
			int new_obstacle_hit_count = this_obstacle_hits[ current_linear_index ] + 1;
			auto [ left_dir_char, right_dir_char ] = LeftRightDirections( greedy_direction );
			auto left_offset = kDirectionDeltaMap.at( left_dir_char );
			auto right_offset = kDirectionDeltaMap.at( right_dir_char );

			int	 block_left_row = greedy_cell_row + left_offset.first;
			int	 block_left_col = greedy_cell_col + left_offset.second;
			int	 block_right_row = greedy_cell_row + right_offset.first;
			int	 block_right_col = greedy_cell_col + right_offset.second;
			bool trigger_rebirth = IsCellBlocked( grid, block_left_row, block_left_col ) && IsCellBlocked( grid, block_right_row, block_right_col );

			if ( trigger_rebirth && !this_rebirth_used[ current_linear_index ] )
			{
				this_rebirth_used[ current_linear_index ] = true;
				this_queue.push_back( current_linear_index );  // 当前节点重入队列
				this_obstacle_hits[ current_linear_index ] = new_obstacle_hit_count;

				// 尝试 C cell（B 再往前一格）
				int c_cell_row = greedy_cell_row + step_row_offset;
				int c_cell_col = greedy_cell_col + step_col_offset;
				if ( IsCellPassable( grid, c_cell_row, c_cell_col ) )
				{
					int c_cell_index = ToIndex( c_cell_row, c_cell_col );
					if ( this_depth[ c_cell_index ] == -1 )
					{
						this_depth[ c_cell_index ] = current_depth_value + 1;
						this_parent[ c_cell_index ] = current_linear_index;
						this_obstacle_hits[ c_cell_index ] = new_obstacle_hit_count;
						this_queue.push_back( c_cell_index );
						CheckMeet( c_cell_index, this_depth, opposite_depth );
					}
				}
			}

			// 3) 首/多次障碍规则
			if ( new_obstacle_hit_count == 1 )
			{
				// 第一次撞墙：尝试其余 3 个方向
				for ( char scan_dir : kDirectionPriorityOrder )
				{
					if ( scan_dir == greedy_direction )
						continue;
					auto delta = kDirectionDeltaMap.at( scan_dir );
					int	 alt_row = current_row + delta.first;
					int	 alt_col = current_col + delta.second;
					if ( IsCellPassable( grid, alt_row, alt_col ) )
					{
						int alt_index = ToIndex( alt_row, alt_col );
						if ( this_depth[ alt_index ] == -1 )
						{
							this_depth[ alt_index ] = current_depth_value + 1;
							this_parent[ alt_index ] = current_linear_index;
							this_obstacle_hits[ alt_index ] = new_obstacle_hit_count;
							this_queue.push_back( alt_index );
							CheckMeet( alt_index, this_depth, opposite_depth );
						}
					}
				}
			}
			else
			{
				// 多次撞墙：先尝试继续往 B 走，否则尝试左右，再尝试反方向
				if ( IsCellPassable( grid, greedy_cell_row, greedy_cell_col ) )
				{
					int b_index = ToIndex( greedy_cell_row, greedy_cell_col );
					if ( this_depth[ b_index ] == -1 )
					{
						this_depth[ b_index ] = current_depth_value + 1;
						this_parent[ b_index ] = current_linear_index;
						this_obstacle_hits[ b_index ] = new_obstacle_hit_count;
						this_queue.push_back( b_index );
						CheckMeet( b_index, this_depth, opposite_depth );
					}
				}
				else
				{
					// 左右两个方向
					for ( char side_dir_char : { left_dir_char, right_dir_char } )
					{
						auto side_delta = kDirectionDeltaMap.at( side_dir_char );
						int	 side_row = current_row + side_delta.first;
						int	 side_col = current_col + side_delta.second;
						if ( IsCellPassable( grid, side_row, side_col ) )
						{
							int side_index = ToIndex( side_row, side_col );
							if ( this_depth[ side_index ] == -1 )
							{
								this_depth[ side_index ] = current_depth_value + 1;
								this_parent[ side_index ] = current_linear_index;
								this_obstacle_hits[ side_index ] = new_obstacle_hit_count;
								this_queue.push_back( side_index );
								CheckMeet( side_index, this_depth, opposite_depth );
							}
						}
					}
					// 反方向
					char opposite_char = OppositeDirection( greedy_direction );
					auto opposite_delta = kDirectionDeltaMap.at( opposite_char );
					int	 opposite_row = current_row + opposite_delta.first;
					int	 opposite_col = current_col + opposite_delta.second;
					if ( IsCellPassable( grid, opposite_row, opposite_col ) )
					{
						int opp_index = ToIndex( opposite_row, opposite_col );
						if ( this_depth[ opp_index ] == -1 )
						{
							this_depth[ opp_index ] = current_depth_value + 1;
							this_parent[ opp_index ] = current_linear_index;
							this_obstacle_hits[ opp_index ] = new_obstacle_hit_count;
							this_queue.push_back( opp_index );
							CheckMeet( opp_index, this_depth, opposite_depth );
						}
					}
				}
			}
		};

		// 主循环
		while ( !frontier_queue_from_start.empty() || !frontier_queue_from_goal.empty() )
		{
			if ( meet_linear_index != -1 )
			{
				int	 max_flush_depth = flush_depth_levels.empty() ? -1 : *flush_depth_levels.rbegin();
				bool start_done = ( frontier_queue_from_start.empty() || depth_from_start[ frontier_queue_from_start.front() ] > max_flush_depth );
				bool goal_done = ( frontier_queue_from_goal.empty() || depth_from_goal[ frontier_queue_from_goal.front() ] > max_flush_depth );
				if ( start_done && goal_done )
					break;
			}

			if ( !frontier_queue_from_start.empty() )
			{
				int current_front_index = frontier_queue_from_start.front();
				frontier_queue_from_start.pop_front();
				ExpandFrontier( current_front_index, goal_pos.row, goal_pos.col, depth_from_start, depth_from_goal, parent_from_start, obstacle_hits_from_start, frontier_queue_from_start, rebirth_used_from_start, 'F' );
			}
			if ( !frontier_queue_from_goal.empty() )
			{
				int current_back_index = frontier_queue_from_goal.front();
				frontier_queue_from_goal.pop_front();
				ExpandFrontier( current_back_index, start_pos.row, start_pos.col, depth_from_goal, depth_from_start, parent_from_goal, obstacle_hits_from_goal, frontier_queue_from_goal, rebirth_used_from_goal, 'R' );
			}
		}

		SearchOutcome outcome;
		outcome.statistics = stats;

		if ( meet_linear_index == -1 )
		{
			outcome.success = false;
			return outcome;
		}

		// 路径重建
		std::vector<int> linear_index_path;
		int				 trace_index = meet_linear_index;
		while ( trace_index != -1 )
		{
			linear_index_path.push_back( trace_index );
			trace_index = parent_from_start[ trace_index ];
		}
		std::reverse( linear_index_path.begin(), linear_index_path.end() );
		trace_index = parent_from_goal[ meet_linear_index ];
		while ( trace_index != -1 )
		{
			linear_index_path.push_back( trace_index );
			trace_index = parent_from_goal[ trace_index ];
		}

		outcome.final_path.reserve( linear_index_path.size() );
		for ( int path_linear_index : linear_index_path )
		{
			outcome.final_path.push_back( FromIndex( path_linear_index ) );
		}

		outcome.meet_position = FromIndex( meet_linear_index );
		outcome.statistics.final_path_length = static_cast<int>( outcome.final_path.size() );
		outcome.success = true;
		return outcome;
	}

	// =========================== 校验 & 打印 ===========================

	inline bool ValidatePathContiguity( const Grid& grid, const std::vector<CellPosition>& path )
	{
		if ( path.empty() )
			return false;
		auto IsOk = [ & ]( const CellPosition& pos ) {
			return pos.row >= 0 && pos.row < static_cast<int>( grid.size() ) && pos.col >= 0 && pos.col < static_cast<int>( grid[ 0 ].size() ) && grid[ pos.row ][ pos.col ] == 0;
		};

		// size_t idx_path (路径中的索引)
		for ( std::size_t idx_path = 0; idx_path < path.size(); ++idx_path )
		{
			if ( !IsOk( path[ idx_path ] ) )
				return false;
			if ( idx_path > 0 )
			{
				int manhattan = std::abs( path[ idx_path ].row - path[ idx_path - 1 ].row ) + std::abs( path[ idx_path ].col - path[ idx_path - 1 ].col );
				if ( manhattan != 1 )
					return false;
			}
		}
		return true;
	}

	inline void RenderGridWithPath( const Grid& grid, const std::vector<CellPosition>& path, CellPosition start_pos, CellPosition goal_pos )
	{
		const std::string& GLYPH_WALL = g_config.use_ascii_glyphs ? g_config.WALL_A : g_config.WALL_U;
		const std::string& GLYPH_EMPTY = g_config.use_ascii_glyphs ? g_config.EMPTY_A : g_config.EMPTY_U;
		const std::string& GLYPH_PATH = g_config.use_ascii_glyphs ? g_config.PATH_A : g_config.PATH_U;
		const std::string& GLYPH_START = g_config.use_ascii_glyphs ? g_config.START_A : g_config.START_U;
		const std::string& GLYPH_END = g_config.use_ascii_glyphs ? g_config.END_A : g_config.END_U;
		const auto&		   GLYPH_ARROW = g_config.use_ascii_glyphs ? g_config.ARROWS_A : g_config.ARROWS_U;

		int grid_height = static_cast<int>( grid.size() );
		int grid_width = static_cast<int>( grid[ 0 ].size() );

		std::unordered_set<long long> path_cell_keys;
		path_cell_keys.reserve( path.size() * 2 );
		auto MakeKey = [ & ]( int row, int col ) -> long long {
			return ( static_cast<long long>( row ) << 32 ) | static_cast<long long>( col );
		};
		for ( const auto& pos : path )
			path_cell_keys.insert( MakeKey( pos.row, pos.col ) );

		std::unordered_map<long long, std::string> arrow_glyph_map;
		if ( g_config.print_arrows && path.size() > 1 )
		{
			for ( std::size_t idx_arrow = 1; idx_arrow < path.size(); ++idx_arrow )
			{
				int prev_row = path[ idx_arrow - 1 ].row;
				int prev_col = path[ idx_arrow - 1 ].col;
				int cur_row = path[ idx_arrow ].row;
				int cur_col = path[ idx_arrow ].col;
				int diff_row = cur_row - prev_row;
				int diff_col = cur_col - prev_col;
				if ( diff_row == -1 )
					arrow_glyph_map[ MakeKey( cur_row, cur_col ) ] = GLYPH_ARROW.at( 'U' );
				else if ( diff_row == 1 )
					arrow_glyph_map[ MakeKey( cur_row, cur_col ) ] = GLYPH_ARROW.at( 'D' );
				else if ( diff_col == -1 )
					arrow_glyph_map[ MakeKey( cur_row, cur_col ) ] = GLYPH_ARROW.at( 'L' );
				else if ( diff_col == 1 )
					arrow_glyph_map[ MakeKey( cur_row, cur_col ) ] = GLYPH_ARROW.at( 'R' );
			}
		}

		for ( int render_row = 0; render_row < grid_height; ++render_row )
		{
			std::ostringstream line_buffer;
			for ( int render_col = 0; render_col < grid_width; ++render_col )
			{
				if ( render_row == start_pos.row && render_col == start_pos.col )
					line_buffer << GLYPH_START;
				else if ( render_row == goal_pos.row && render_col == goal_pos.col )
					line_buffer << GLYPH_END;
				else if ( grid[ render_row ][ render_col ] == 1 )
					line_buffer << GLYPH_WALL;
				else
				{
					long long key = MakeKey( render_row, render_col );
					if ( path_cell_keys.count( key ) )
					{
						if ( g_config.print_arrows && arrow_glyph_map.count( key ) )
							line_buffer << arrow_glyph_map[ key ];
						else
							line_buffer << GLYPH_PATH;
					}
					else
					{
						line_buffer << GLYPH_EMPTY;
					}
				}
			}
			std::cout << line_buffer.str() << "\n";
		}
	}

	// =========================== 命令行解析 ===========================

	struct ArgStream
	{
		std::vector<std::string> tokens;
		std::size_t				 cursor = 0;

		bool HasNext() const
		{
			return cursor < tokens.size();
		}
		std::string Get()
		{
			return cursor < tokens.size() ? tokens[ cursor++ ] : "";
		}
		std::string Peek() const
		{
			return cursor < tokens.size() ? tokens[ cursor ] : "";
		}
	};

	inline void ParseArgs( int argc, char** argv )
	{
		ArgStream arg_stream;
		arg_stream.tokens.assign( argv + 1, argv + argc );

		auto ExpectValue = [ & ]( const std::string& flag ) -> std::string {
			if ( !arg_stream.HasNext() )
				throw std::runtime_error( "Missing value after " + flag );
			return arg_stream.Get();
		};

		while ( arg_stream.HasNext() )
		{
			std::string token = arg_stream.Get();
			if ( token == "--map" )
			{
				g_config.map_file_path = ExpectValue( token );
				g_config.use_random_map = false;
			}
			else if ( token == "--sx" )
			{
				int value = std::stoi( ExpectValue( token ) );
				if ( !g_config.cli_start_override )
					g_config.cli_start_override = CellPosition { -1, -1 };
				g_config.cli_start_override->row = value;
			}
			else if ( token == "--sy" )
			{
				int value = std::stoi( ExpectValue( token ) );
				if ( !g_config.cli_start_override )
					g_config.cli_start_override = CellPosition { -1, -1 };
				g_config.cli_start_override->col = value;
			}
			else if ( token == "--ex" )
			{
				int value = std::stoi( ExpectValue( token ) );
				if ( !g_config.cli_goal_override )
					g_config.cli_goal_override = CellPosition { -1, -1 };
				g_config.cli_goal_override->row = value;
			}
			else if ( token == "--ey" )
			{
				int value = std::stoi( ExpectValue( token ) );
				if ( !g_config.cli_goal_override )
					g_config.cli_goal_override = CellPosition { -1, -1 };
				g_config.cli_goal_override->col = value;
			}
			else if ( token == "--random" )
			{
				// 需要 3 个值：W H P
				if ( arg_stream.cursor + 2 >= arg_stream.tokens.size() )
					throw std::runtime_error( "--random W H P requires 3 values" );
				g_config.use_random_map = true;
				g_config.random_map_width = std::stoi( arg_stream.Get() );
				g_config.random_map_height = std::stoi( arg_stream.Get() );
				g_config.random_wall_probability = std::stod( arg_stream.Get() );
			}
			else if ( token == "--seed" )
			{
				g_config.random_seed = static_cast<std::uint32_t>( std::stoul( ExpectValue( token ) ) );
			}
			else if ( token == "--wait" )
			{
				g_config.flush_wait_layers = std::stoi( ExpectValue( token ) );
			}
			else if ( token == "--no-print" )
			{
				g_config.print_path = false;
			}
			else if ( token == "--arrow" )
			{
				g_config.print_arrows = true;
			}
			else if ( token == "--ascii" )
			{
				g_config.use_ascii_glyphs = true;
			}
			else if ( token == "--no-ensure" )
			{
				g_config.reroll_until_solvable = false;
			}
			else if ( token == "--max-try" )
			{
				g_config.reroll_max_attempts = std::stoi( ExpectValue( token ) );
			}
			else
			{
				std::cerr << "Unknown option: " << token << "\n";
				std::exit( 1 );
			}
		}
	}

}  // namespace ibpbstar

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
			return 0;
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
		return 0;
	}
}

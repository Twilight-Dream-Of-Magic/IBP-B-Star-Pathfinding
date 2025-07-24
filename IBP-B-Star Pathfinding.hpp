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

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <queue>
#include <vector>
#include <array>
#include <stack>
#include <set>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <algorithm>
#include <random>

struct CellPosition
{
	int row;
	int col;

	bool operator==( const CellPosition& other ) const
	{
		return row == other.row && col == other.col;
	}
};

namespace std
{
	template <>
	struct hash<CellPosition>
	{
		size_t operator()( const CellPosition& p ) const
		{
			return hash<int>()( p.row ) ^ ( hash<int>()( p.col ) << 1 );
		}
	};
}  // namespace std

namespace IBP_BStarAlogithm
{

	// =========================== 基础类型 ===========================

	using Grid = std::vector<std::vector<int>>;

	struct RunConfig
	{
		// 地图来源配置
		bool						use_random_map = true;
		int							random_map_width = 64;
		int							random_map_height = 64;
		double						random_wall_probability = 0.20;
		std::uint32_t				random_seed = 56464641u;
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

	inline bool IsConcaveEntry( const Grid& grid, int base_row, int base_col, char greedy_direction, int grid_height, int grid_width )
	{
		if ( greedy_direction == 'L' || greedy_direction == 'R' )
		{
			bool cell_up_free = false;
			bool cell_down_free = false;

			// 上方检查
			if ( base_row > 0 )
			{
				cell_up_free = IsCellPassable( grid, base_row - 1, base_col );
			}

			// 下方检查
			if ( base_row < grid_height - 1 )
			{
				cell_down_free = IsCellPassable( grid, base_row + 1, base_col );
			}

			return cell_up_free && cell_down_free;
		}
		else
		{
			bool cell_left_free = false;
			bool cell_right_free = false;

			// 左侧检查
			if ( base_col > 0 )
			{
				cell_left_free = IsCellPassable( grid, base_row, base_col - 1 );
			}

			// 右侧检查
			if ( base_col < grid_width - 1 )
			{
				cell_right_free = IsCellPassable( grid, base_row, base_col + 1 );
			}

			return cell_left_free && cell_right_free;
		}
	}

	// =========================== 地图生成 / 读取 ===========================

	inline Grid GenerateRandomGrid( int width, int height, double wall_probability, std::uint32_t seed, CellPosition& out_start, CellPosition& out_goal )
	{
		LinearCongruentialRng32 prng( seed );
		Grid					grid( height, std::vector<int>( width, 0 ) );
		for ( int row_index = 0; row_index < height; ++row_index )
		{
			for ( int col_index = 0; col_index < width; ++col_index )
			{
				if ( prng.NextFloatZeroToOne() < wall_probability )
					grid[ row_index ][ col_index ] = 1;
			}
		}
		out_start = { 0, 0 };
		out_goal = { height - 1, width - 1 };
		grid[ out_start.row ][ out_start.col ] = 0;
		grid[ out_goal.row ][ out_goal.col ] = 0;
		return grid;
	}

	// 并查集实现
	class UnionFind
	{
	private:
		std::vector<int> parent;
		std::vector<int> rank;
		int				 num_sets;

	public:
		UnionFind( int n ) : parent( n ), rank( n, 0 ), num_sets( n )
		{
			for ( int i = 0; i < n; ++i )
			{
				parent[ i ] = i;
			}
		}

		int find( int x )
		{
			if ( parent[ x ] != x )
			{
				parent[ x ] = find( parent[ x ] );
			}
			return parent[ x ];
		}

		void unite( int x, int y )
		{
			int rx = find( x );
			int ry = find( y );
			if ( rx == ry )
				return;

			if ( rank[ rx ] < rank[ ry ] )
			{
				parent[ rx ] = ry;
			}
			else if ( rank[ rx ] > rank[ ry ] )
			{
				parent[ ry ] = rx;
			}
			else
			{
				parent[ ry ] = rx;
				rank[ rx ]++;
			}
			num_sets--;
		}

		bool connected( int x, int y )
		{
			return find( x ) == find( y );
		}

		int getNumSets() const
		{
			return num_sets;
		}
	};

	inline Grid GeneratePerfectMazeGrid( int width, int height, std::uint32_t seed, CellPosition& out_start, CellPosition& out_goal )
	{
		// 初始化随机数引擎
		std::mt19937 prng( seed );

		if ( width < 3 || height < 3 )
		{
			throw std::runtime_error( "Maze size too small" );
		}

		const int cell_width = ( width - 1 ) / 2;	 // 实际单元格列数
		const int cell_height = ( height - 1 ) / 2;	 // 实际单元格行数
		const int total_cells = cell_width * cell_height;

		// 单单元格迷宫特殊处理
		if ( total_cells == 1 )
		{
			Grid grid( height, std::vector<int>( width, 1 ) );
			grid[ 1 ][ 1 ] = 0;
			out_start = { 1, 1 };
			out_goal = { 1, 1 };
			return grid;
		}

		// 1. 初始化网格：全部设为墙
		Grid grid( height, std::vector<int>( width, 1 ) );

		// 2. 初始化并查集
		UnionFind union_find_set( total_cells );

		// 3. 生成所有可拆的墙（单元格之间的墙）
		struct Wall
		{
			CellPosition wall_pos;	 // 墙的位置
			int			 cell1_idx;	 // 墙的一侧单元格索引
			int			 cell2_idx;	 // 墙的另一侧单元格索引
		};
		std::vector<Wall> walls;

		// 辅助函数：将单元格坐标（行、列）转换为线性索引
		auto cellToIndex = [ cell_width ]( int cell_row, int cell_col ) {
			return cell_row * cell_width + cell_col;
		};

		// 辅助函数：将网格坐标转换为单元格坐标
		auto gridToCell = []( int grid_r, int grid_c ) -> std::pair<int, int> {
			return { ( grid_r - 1 ) / 2, ( grid_c - 1 ) / 2 };
		};

		// 生成水平墙（在单元格行之间）
		for ( int cell_r = 0; cell_r < cell_height - 1; ++cell_r )
		{
			for ( int cell_c = 0; cell_c < cell_width; ++cell_c )
			{
				Wall w;
				w.wall_pos = { cell_r * 2 + 2, cell_c * 2 + 1 };  // 墙的位置
				w.cell1_idx = cellToIndex( cell_r, cell_c );	  // 上方单元格
				w.cell2_idx = cellToIndex( cell_r + 1, cell_c );  // 下方单元格
				walls.push_back( w );
			}
		}

		// 生成垂直墙（在单元格列之间）
		for ( int cell_r = 0; cell_r < cell_height; ++cell_r )
		{
			for ( int cell_c = 0; cell_c < cell_width - 1; ++cell_c )
			{
				Wall w;
				w.wall_pos = { cell_r * 2 + 1, cell_c * 2 + 2 };  // 墙的位置
				w.cell1_idx = cellToIndex( cell_r, cell_c );	  // 左侧单元格
				w.cell2_idx = cellToIndex( cell_r, cell_c + 1 );  // 右侧单元格
				walls.push_back( w );
			}
		}

		// 4. 随机打乱墙
		std::shuffle( walls.begin(), walls.end(), prng );

		// 5. 按随机顺序处理每一面墙
		for ( const Wall& w : walls )
		{
			// Kruskal Alogrithm
			// 如果墙两侧的单元格属于不同集合，则打通这面墙
			if ( !union_find_set.connected( w.cell1_idx, w.cell2_idx ) )
			{
				union_find_set.unite( w.cell1_idx, w.cell2_idx );

				// 打通墙
				grid[ w.wall_pos.row ][ w.wall_pos.col ] = 0;

				// 打通墙两侧的单元格（确保单元格本身是通路）
				auto [ cell1_r, cell1_c ] = gridToCell( w.wall_pos.row, w.wall_pos.col );
				grid[ cell1_r * 2 + 1 ][ cell1_c * 2 + 1 ] = 0;

				auto [ cell2_r, cell2_c ] = gridToCell( w.wall_pos.row + ( w.wall_pos.row % 2 == 0 ? 1 : 0 ), w.wall_pos.col + ( w.wall_pos.col % 2 == 0 ? 1 : 0 ) );
				grid[ cell2_r * 2 + 1 ][ cell2_c * 2 + 1 ] = 0;
			}
		}

		// 6. 确保所有单元格都是通路（处理边缘情况）
		for ( int cell_r = 0; cell_r < cell_height; ++cell_r )
		{
			for ( int cell_c = 0; cell_c < cell_width; ++cell_c )
			{
				grid[ cell_r * 2 + 1 ][ cell_c * 2 + 1 ] = 0;
			}
		}

		// 7. 使用双BFS寻找最长路径的起点和终点
		constexpr int dx[ 4 ] = { 0, 1, 0, -1 };
		constexpr int dy[ 4 ] = { -1, 0, 1, 0 };

		auto inBounds = [ & ]( int r, int c ) {
			return r >= 0 && r < height && c >= 0 && c < width;
		};

		// BFS函数，返回从起点出发的最长路径长度和最远点
		auto bfs = [ & ]( CellPosition start, CellPosition& farthest ) -> int {
			std::vector<int>		 distances( height * width, -1 );
			std::queue<CellPosition> position_queue;
			position_queue.push( start );
			distances[ start.row * width + start.col ] = 0;
			farthest = start;
			int max_dist = 0;

			while ( !position_queue.empty() )
			{
				CellPosition current_position = position_queue.front();
				position_queue.pop();
				int cur_idx = current_position.row * width + current_position.col;

				if ( distances[ cur_idx ] > max_dist )
				{
					max_dist = distances[ cur_idx ];
					farthest = current_position;
				}

				for ( int d = 0; d < 4; ++d )
				{
					int nr = current_position.row + dy[ d ];
					int nc = current_position.col + dx[ d ];
					if ( inBounds( nr, nc ) )
					{
						int next_idx = nr * width + nc;
						if ( grid[ nr ][ nc ] == 0 && distances[ next_idx ] == -1 )
						{
							distances[ next_idx ] = distances[ cur_idx ] + 1;
							position_queue.push( { nr, nc } );
						}
					}
				}
			}
			return max_dist;
		};

		// 选择一个随机的通路位置作为BFS起点
		std::vector<CellPosition> open_cells;
		for ( int r = 1; r < height; r += 2 )
		{
			for ( int c = 1; c < width; c += 2 )
			{
				if ( grid[ r ][ c ] == 0 )
				{
					open_cells.push_back( { r, c } );
				}
			}
		}

		if ( open_cells.empty() )
		{
			throw std::runtime_error( "No open cell found" );
		}

		std::uniform_int_distribution<int> dist( 0, open_cells.size() - 1 );
		CellPosition					   start_bfs = open_cells[ dist( prng ) ];

		CellPosition pointA;
		bfs( start_bfs, pointA );

		CellPosition pointB;
		int			 path_length = bfs( pointA, pointB );

		// 8. 设置起点和终点
		out_start = pointA;
		out_goal = pointB;

		// 9. 确保起点和终点标记为通路（冗余安全）
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

	struct ObstacleHistory
	{
		int	 hit_count = 0;			// 碰撞次数
		char last_direction = 0;	// 最后碰撞方向
		bool rebirth_used = false;	// 再生机制是否使用
		bool zigzag_used = false;	// 弯延绕模式是否已尝试
	};

	struct SearchStatistics
	{
		int expanded_node_count = 0;
		int final_path_length = 0;
		int zigzag_activations = 0;	 // 弯延绕激活次数
		int zigzag_successes = 0;	 // 弯延绕成功扩展次数
	};

	struct SearchOutcome
	{
		std::vector<CellPosition> final_path;
		CellPosition			  meet_position { -1, -1 };
		SearchStatistics		  statistics;
		bool					  success = false;
	};

	constexpr uint32_t ZigzagModeThresholds = 8;
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

		NodeQueue					 frontier_queue_from_start;
		NodeQueue					 frontier_queue_from_goal;
		std::vector<ObstacleHistory> history_from_start( total_cells );
		std::vector<ObstacleHistory> history_from_goal( total_cells );

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


		// =========================== 修复相遇检测 ===========================
		auto CheckMeet = [ & ]( int node_linear_index, DepthArray& this_side_depth, DepthArray& other_side_depth ) -> void {
			// 修复1：添加更宽松的相遇条件
			bool hit_terminal = ( node_linear_index == goal_linear_index ) || ( node_linear_index == start_linear_index );

			// 修复2：检查是否在对方已访问节点中
			bool hit_other_side = ( other_side_depth[ node_linear_index ] != -1 );

			if ( ( hit_terminal || hit_other_side ) && meet_linear_index == -1 )
			{
				meet_linear_index = node_linear_index;
				meet_depth_sum = this_side_depth[ node_linear_index ] + ( other_side_depth[ node_linear_index ] == -1 ? 0 : other_side_depth[ node_linear_index ] );

				// 修复3：确保等待层数合理
				flush_depth_levels.clear();
				flush_depth_levels.insert( meet_depth_sum );
				for ( int wait_layer = 1; wait_layer <= wait_layers; ++wait_layer )
				{
					if ( meet_depth_sum + wait_layer < total_cells )
					{
						flush_depth_levels.insert( meet_depth_sum + wait_layer );
					}
					if ( meet_depth_sum - wait_layer >= 0 )
					{
						flush_depth_levels.insert( meet_depth_sum - wait_layer );
					}
				}
			}
		};

		// ====================== 弯延绕模式实现 ======================
		std::vector<std::vector<int>> zigzag_marks( grid_height, std::vector<int>( grid_width, 0 ) );
		auto						  RunZigzagMode = [ & ]( int row, int col, char direction, int obstacle_hit_count, DepthArray& depth_arr, ParentArray& parent_arr, NodeQueue& queue, std::vector<ObstacleHistory>& history ) -> bool {
			auto [ left_dir, right_dir ] = LeftRightDirections( direction );
			auto left_offset = kDirectionDeltaMap.at( left_dir );
			auto right_offset = kDirectionDeltaMap.at( right_dir );

			// 基础左右扩展
			std::vector<std::pair<int, int>> zigzag_moves = {
				{ row + left_offset.first, col + left_offset.second },	  // 左/右方向
				{ row + right_offset.first, col + right_offset.second }  // 右/左方向
			};

			// 高级弯延绕：尝试对角线方向 (适用于多次碰撞)
			if ( obstacle_hit_count > ZigzagModeThresholds )
			{
				// 获取对角线方向 (4个方向迷宫的对角线模拟)
				std::vector<char> diagonal_dirs;
				if ( direction == 'U' || direction == 'D' )
				{
					 diagonal_dirs = { 'L', 'R' };
				}
				else
				{
					 diagonal_dirs = { 'U', 'D' };
				}

				for ( char diag_dir : diagonal_dirs )
				{
					auto diag_offset = kDirectionDeltaMap.at( diag_dir );
					zigzag_moves.push_back( { row + diag_offset.first, col + diag_offset.second } );
				}
			}

			bool expanded = false;
			for ( const auto& [ new_row, new_col ] : zigzag_moves )
			{
				if ( IsCellPassable( grid, new_row, new_col ) )
				{
					int new_index = ToIndex( new_row, new_col );
					if ( depth_arr[ new_index ] == -1 )
					{
						depth_arr[ new_index ] = depth_arr[ ToIndex( row, col ) ] + 1;
						parent_arr[ new_index ] = ToIndex( row, col );

						// 更新历史记录 - 添加弯延绕标记
						ObstacleHistory new_history = history[ ToIndex( row, col ) ];
						new_history.hit_count++;
						new_history.last_direction = direction;
						new_history.zigzag_used = true;  // 标记为弯延绕扩展
						history[ new_index ] = new_history;

						queue.push_back( new_index );
						CheckMeet( new_index, depth_arr, ( &depth_arr == &depth_from_start ) ? depth_from_goal : depth_from_start );
						expanded = true;
					}
				}
			}

			if constexpr ( false )
			{
				if ( expanded )
				{
					stats.zigzag_activations++;
					std::cout << "Zigzag activated at (" << row << "," << col << ") hit_count=" << obstacle_hit_count << "\n";
				}
			}

			return expanded;
		};

		// 核心修复：正确的 ExpandFrontier 实现
		auto ExpandFrontier = [ & ]( int current_linear_index, int greedy_goal_row, int greedy_goal_col, DepthArray& this_depth, DepthArray& opposite_depth, ParentArray& this_parent, NodeQueue& this_queue, std::vector<ObstacleHistory>& this_history, char debug_tag ) {
			stats.expanded_node_count++;

			//当前格子 (A格)
			CellPosition	 current_pos = FromIndex( current_linear_index );
			int				 current_row = current_pos.row;
			int				 current_col = current_pos.col;
			ObstacleHistory& current_history = this_history[ current_linear_index ];

			// 1. 计算贪心方向 (论文章节 4.1)
			char greedy_direction = ChooseGreedyDirection( current_row, current_col, greedy_goal_row, greedy_goal_col );
			auto direction_offset = kDirectionDeltaMap.at( greedy_direction );
			int	 step_row_offset = direction_offset.first;
			int	 step_col_offset = direction_offset.second;

			// 2. 目标格子（B格）(C 格)
			//greedy_cell
			//next_greedy_cell
			int B_row = current_row + step_row_offset;
			int B_col = current_col + step_col_offset;
			int C_row = current_row + 2 * step_row_offset;
			int C_col = current_col + 2 * step_col_offset;
			int C_index = ToIndex( C_row, C_col );
			int B_index = ToIndex( B_row, B_col );

			// 3. 预探索机制 (论文章节 4.3) (贪心思路 A->C)
			if ( IsCellPassable( grid, B_row, B_col ) && IsCellPassable( grid, C_row, C_col ) )
			{
				bool is_concave = IsConcaveEntry( grid, B_row, B_col, greedy_direction, grid_height, grid_width );
				bool is_next_concave = IsConcaveEntry( grid, C_row, C_col, greedy_direction, grid_height, grid_width );

				// 如果是凹形陷阱，提前扩展其他方向
				if ( is_concave || is_next_concave )
				{
					for ( char dir : kDirectionPriorityOrder )
					{
						if ( dir == greedy_direction )
							continue;

						auto dir_offset = kDirectionDeltaMap.at( dir );
						int	 move_to_row = current_row + dir_offset.first;
						int	 move_to_col = current_col + dir_offset.second;

						if ( IsCellPassable( grid, move_to_row, move_to_col ) )
						{
							int new_index = ToIndex( move_to_row, move_to_col );
							if ( this_depth[ new_index ] == -1 )
							{
								this_depth[ new_index ] = this_depth[ current_linear_index ] + 1;
								this_parent[ new_index ] = current_linear_index;
								this_history[ new_index ] = current_history;  // 复制历史
								this_queue.push_back( new_index );
								CheckMeet( new_index, this_depth, opposite_depth );
							}
						}
					}
				}
			}

			// 4. 判断 B 格是否可走 (贪心思路 A->B)
			if ( IsCellPassable( grid, B_row, B_col ) )
			{
				if ( this_depth[ B_index ] == -1 )
				{
					this_depth[ B_index ] = this_depth[ current_linear_index ] + 1;
					this_parent[ B_index ] = current_linear_index;
					this_history[ B_index ] = current_history;
					this_queue.push_back( B_index );
					CheckMeet( B_index, this_depth, opposite_depth );
				}
				return;	 // 一步走 B 后结束本次扩展
			}

			// 5. 障碍处理 (论文章节 4.1 和 4.2)
			current_history.hit_count++;
			current_history.last_direction = greedy_direction;

			// 6. 障碍规避再生机制 (论文章节 4.2)
			auto [ left_dir, right_dir ] = LeftRightDirections( greedy_direction );
			auto left_offset = kDirectionDeltaMap.at( left_dir );
			auto right_offset = kDirectionDeltaMap.at( right_dir );

			int block_left_row = B_row + left_offset.first;
			int block_left_col = B_col + left_offset.second;
			int block_right_row = B_row + right_offset.first;
			int block_right_col = B_col + right_offset.second;

			bool left_blocked = IsCellBlocked( grid, block_left_row, block_left_col );
			bool right_blocked = IsCellBlocked( grid, block_right_row, block_right_col );

			// 严格障碍条件（左右都是障碍）
			if ( left_blocked && right_blocked && !current_history.rebirth_used )
			{
				current_history.rebirth_used = true;

				// 碰撞计数后重入队列
				current_history.hit_count++;
				this_queue.push_back( current_linear_index );

				// 垂直方向扩展
				auto [ alt_dir1, alt_dir2 ] = ( greedy_direction == 'U' || greedy_direction == 'D' ) ? std::make_pair( 'L', 'R' ) : std::make_pair( 'U', 'D' );

				for ( char alt_dir : { alt_dir1, alt_dir2 } )
				{
					auto alt_offset = kDirectionDeltaMap.at( alt_dir );
					int	 alt_row = current_row + alt_offset.first;
					int	 alt_col = current_col + alt_offset.second;

					if ( IsCellPassable( grid, alt_row, alt_col ) )
					{
						int alt_index = ToIndex( alt_row, alt_col );
						if ( this_depth[ alt_index ] == -1 )
						{
							this_depth[ alt_index ] = this_depth[ current_linear_index ] + 1;
							this_parent[ alt_index ] = current_linear_index;
							this_history[ alt_index ] = current_history;
							this_queue.push_back( alt_index );
							CheckMeet( alt_index, this_depth, opposite_depth );
						}
					}
				}
				return;
			}

			// 7. 根据碰撞次数处理
			if ( current_history.hit_count == 1 )
			{
				// 第一次碰撞：尝试其他三个方向 (论文图1示例)
				for ( char dir : kDirectionPriorityOrder )
				{
					if ( dir == greedy_direction )
						continue;

					auto dir_offset = kDirectionDeltaMap.at( dir );
					int	 new_row = current_row + dir_offset.first;
					int	 new_col = current_col + dir_offset.second;

					if ( IsCellPassable( grid, new_row, new_col ) )
					{
						int new_index = ToIndex( new_row, new_col );
						if ( this_depth[ new_index ] == -1 )
						{
							this_depth[ new_index ] = this_depth[ current_linear_index ] + 1;
							this_parent[ new_index ] = current_linear_index;
							this_history[ new_index ] = current_history;
							this_queue.push_back( new_index );
							CheckMeet( new_index, this_depth, opposite_depth );
						}
					}
				}
			}
			else
			{
				// 多次碰撞：根据障碍状态决策 (论文章节 4.1)
				// 检查原障碍是否仍然存在
				bool original_obstacle_exists = IsCellBlocked( grid, current_row + direction_offset.first, current_col + direction_offset.second );

				if ( original_obstacle_exists )
				{
					// 障碍仍在：向反方向移动
					char opposite_dir = OppositeDirection( greedy_direction );
					auto opp_offset = kDirectionDeltaMap.at( opposite_dir );
					int	 opp_row = current_row + opp_offset.first;
					int	 opp_col = current_col + opp_offset.second;

					if ( IsCellPassable( grid, opp_row, opp_col ) )
					{
						int opp_index = ToIndex( opp_row, opp_col );
						if ( this_depth[ opp_index ] == -1 )
						{
							this_depth[ opp_index ] = this_depth[ current_linear_index ] + 1;
							this_parent[ opp_index ] = current_linear_index;
							this_history[ opp_index ] = current_history;
							this_queue.push_back( opp_index );
							CheckMeet( opp_index, this_depth, opposite_depth );
						}
					}
				}
				else
				{
					// 障碍消失：尝试原方向
					int orig_row = current_row + direction_offset.first;
					int orig_col = current_col + direction_offset.second;

					if ( IsCellPassable( grid, orig_row, orig_col ) )
					{
						int orig_index = ToIndex( orig_row, orig_col );
						if ( this_depth[ orig_index ] == -1 )
						{
							this_depth[ orig_index ] = this_depth[ current_linear_index ] + 1;
							this_parent[ orig_index ] = current_linear_index;
							this_history[ orig_index ] = current_history;
							this_queue.push_back( orig_index );
							CheckMeet( orig_index, this_depth, opposite_depth );
						}
					}
				}

				if ( current_history.hit_count > ZigzagModeThresholds && !current_history.zigzag_used )
				{
					bool zigzag_expanded = RunZigzagMode( current_row, current_col, greedy_direction, current_history.hit_count, this_depth, this_parent, this_queue, this_history );

					// 如果弯延绕成功扩展，跳过其他障碍处理
					if ( zigzag_expanded )
					{
						return;
					}
					// 标记当前节点已尝试弯延绕
					current_history.zigzag_used = true;
				}
				else if(current_history.hit_count >= ZigzagModeThresholds * 2 && current_history.hit_count % ZigzagModeThresholds == 0)
				{
					for(auto& selected_history : this_history)
					{
						if(selected_history.zigzag_used)
							selected_history.zigzag_used = false;
					}
				}
			}
		};

		auto NeedFlush = [ & ]( const NodeQueue& q, const DepthArray& depth_arr ) -> bool {
			if ( q.empty() )
				return false;
			int front_depth = depth_arr[ q.front() ];
			return flush_depth_levels.count( front_depth ) > 0;
		};

		// 主循环
		while ( !frontier_queue_from_start.empty() || !frontier_queue_from_goal.empty() )
		{
			if constexpr ( false )
			{
				if ( meet_linear_index == -1 )
				{
					std::cout << "[loop] S=" << frontier_queue_from_start.size() << " G=" << frontier_queue_from_goal.size() << "\n";
				}
				else
				{
					std::cout << "[flush] Sneed=" << NeedFlush( frontier_queue_from_start, depth_from_start ) << " Gneed=" << NeedFlush( frontier_queue_from_goal, depth_from_goal ) << " Ssize=" << frontier_queue_from_start.size() << " Gsize=" << frontier_queue_from_goal.size() << "\n";
				}
			}

			// 确保即使未相遇也继续搜索
			if ( meet_linear_index != -1 )
			{
				bool need_start = NeedFlush( frontier_queue_from_start, depth_from_start );
				bool need_goal = NeedFlush( frontier_queue_from_goal, depth_from_goal );

				if ( !need_start && !need_goal )
					break;

				if ( need_start )
				{
					int current_index = frontier_queue_from_start.front();
					frontier_queue_from_start.pop_front();
					ExpandFrontier( current_index, goal_pos.row, goal_pos.col, depth_from_start, depth_from_goal, parent_from_start, frontier_queue_from_start, history_from_start, 'F' );
				}

				if ( need_goal )
				{
					int current_index = frontier_queue_from_goal.front();
					frontier_queue_from_goal.pop_front();
					ExpandFrontier( current_index, start_pos.row, start_pos.col, depth_from_goal, depth_from_start, parent_from_goal, frontier_queue_from_goal, history_from_goal, 'R' );
				}
			}
			else
			{
				// 优化队列处理顺序
				bool start_has_nodes = !frontier_queue_from_start.empty();
				bool goal_has_nodes = !frontier_queue_from_goal.empty();

				// 优先处理节点数较少的队列
				if ( start_has_nodes && ( !goal_has_nodes || frontier_queue_from_start.size() <= frontier_queue_from_goal.size() ) )
				{
					int current_index = frontier_queue_from_start.front();
					frontier_queue_from_start.pop_front();
					ExpandFrontier( current_index, goal_pos.row, goal_pos.col, depth_from_start, depth_from_goal, parent_from_start, frontier_queue_from_start, history_from_start, 'F' );
				}
				else if ( goal_has_nodes )
				{
					int current_index = frontier_queue_from_goal.front();
					frontier_queue_from_goal.pop_front();
					ExpandFrontier( current_index, start_pos.row, start_pos.col, depth_from_goal, depth_from_start, parent_from_goal, frontier_queue_from_goal, history_from_goal, 'R' );
				}
			}

			// 正常阶段处理
			if ( !frontier_queue_from_start.empty() )
			{
				int current_index = frontier_queue_from_start.front();
				frontier_queue_from_start.pop_front();
				ExpandFrontier( current_index, goal_pos.row, goal_pos.col, depth_from_start, depth_from_goal, parent_from_start, frontier_queue_from_start, history_from_start, 'F' );
			}

			if ( !frontier_queue_from_goal.empty() )
			{
				int current_index = frontier_queue_from_goal.front();
				frontier_queue_from_goal.pop_front();
				ExpandFrontier( current_index, start_pos.row, start_pos.col, depth_from_goal, depth_from_start, parent_from_goal, frontier_queue_from_goal, history_from_goal, 'R' );
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

	inline int ShortestPathLength_BFS( const Grid& grid, CellPosition start, CellPosition goal )
	{
		const int height = static_cast<int>( grid.size() );
		const int width = static_cast<int>( grid[ 0 ].size() );

		auto Inside = [ & ]( int r, int c ) {
			return r >= 0 && r < height && c >= 0 && c < width;
		};
		auto ToIndex = [ & ]( int r, int c ) {
			return r * width + c;
		};

		std::vector<int>		 distance( width * height, -1 );
		std::queue<CellPosition> queue;

		if ( grid[ start.row ][ start.col ] != 0 || grid[ goal.row ][ goal.col ] != 0 )
			return -1;

		distance[ ToIndex( start.row, start.col ) ] = 0;
		queue.push( start );

		const int drow[ 4 ] = { -1, 1, 0, 0 };
		const int dcol[ 4 ] = { 0, 0, -1, 1 };

		while ( !queue.empty() )
		{
			CellPosition cur = queue.front();
			queue.pop();
			if ( cur.row == goal.row && cur.col == goal.col )
				return distance[ ToIndex( cur.row, cur.col ) ];

			for ( int k = 0; k < 4; ++k )
			{
				int nr = cur.row + drow[ k ];
				int nc = cur.col + dcol[ k ];
				if ( !Inside( nr, nc ) )
					continue;
				if ( grid[ nr ][ nc ] != 0 )
					continue;
				int idx = ToIndex( nr, nc );
				if ( distance[ idx ] != -1 )
					continue;
				distance[ idx ] = distance[ ToIndex( cur.row, cur.col ) ] + 1;
				queue.push( { nr, nc } );
			}
		}
		return -1;
	}

	// ========= 迷宫演示：额外跑一次（无任何新参数） =========
	inline void MazeDemo()
	{
		CellPosition maze_start, maze_goal;
		Grid		 maze_grid = GeneratePerfectMazeGrid( g_config.random_map_width, g_config.random_map_height,
														  g_config.random_seed ^ 0xA53E1234u,  // 换成 32bit 常量
														  maze_start, maze_goal );

		// ① 先做基准 BFS
		int bfs_len = ShortestPathLength_BFS( maze_grid, maze_start, maze_goal );
		if ( bfs_len == -1 )
		{
			std::cout << "[maze] UNSOLVABLE MAZE (BFS could not reach E)\n";
			RenderGridWithPath( maze_grid, {}, maze_start, maze_goal );
			return;
		}

		// ② 再跑 IBP‑B*
		SearchOutcome maze_out = RunIbpBStar( maze_grid, maze_start, maze_goal, g_config.flush_wait_layers );

		// ③ 输出与校验（路径空要单独处理）
		if ( maze_out.final_path.empty() )
		{
			std::cout << "[maze] NO PATH FOUND by IBP-B*\n";
		}
		else if ( !ValidatePathContiguity( maze_grid, maze_out.final_path ) )
		{
			std::cout << "[maze] INVALID PATH (broken chain)\n";
		}

		if ( g_config.print_stats )
		{
			std::cout << "\n[maze] path_length=" << maze_out.statistics.final_path_length << " meet=(" << maze_out.meet_position.row << ", " << maze_out.meet_position.col << ")"
					  << " expanded=" << maze_out.statistics.expanded_node_count << "\n";
			std::cout << "[maze] bfs_search_length=" << bfs_len << "\n";
		}
		if ( g_config.print_path )
		{
			std::cout << "\n[maze] =====================\n";
			RenderGridWithPath( maze_grid, maze_out.final_path, maze_start, maze_goal );
		}
	}
	// ========= 迷宫演示结束 =========
}  // namespace IBP_BStarAlogithm
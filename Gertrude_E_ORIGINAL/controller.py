from collections.abc import Callable, Iterable
from typing import Union, List, Optional, Tuple, Dict
from collections import deque
import time

from game import *
# from .player_board import PlayerBoard
import random


class PlayerController:
	"""
	You may add functions, however, __init__, bid, and play are the entry
	points for your program and should not be changed.
	"""
	
	def __init__(self, player_parity:int, time_left: Callable):
		# return
		self.player_parity = player_parity
		
		## bfs path: map target location -> next step direction
		self._bfs_cache: Dict[Location, Dict[Location, int]] = {}
		self._bfs_cache_turn: int = -1
		self.transposition_table: Dict = {}
		
	
	def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
		"""
		Called at the start of each round. Return the number of stamina you
		want to bid for initiative. Defaults to zero.
		"""
		player = board.get_player(player_parity)
		opponent = board.get_opponent(player_parity)

		## bid more aggressively when we're stamina rich vs opponent
		if player.stamina > opponent.stamina + 20:
			return 5
		return 3
	
	def play(
		self,
		board: Board,
		player_parity: int,
		time_left: Callable,
	) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:
		"""
		Return either a single Action or an iterable of Actions for this turn.
		This sample agent idles by returning an empty list.
		"""

		## refresh bfs cache each turn
		if board.turn_count != self._bfs_cache_turn:
			self._bfs_cache.clear()
			self._bfs_cache_turn = board.turn_count

		## clear transposition table
		if board.turn_count % 40 == 0:
			self.transposition_table.clear()
		
		## time management - estimate turns remaining for us
		timeleft = time_left()
		turns_remaining = max(1, (GameConstants.MAX_TURNS - board.turn_count) // 2)
		time_budget = min(timeleft * 0.80 / turns_remaining, 1.5)
		time_budget = max(time_budget, 0.04)

		start_time = time.time()

		## generate top level candidates
		candidates = self._get_candidates(board, player_parity)
		if not candidates:
			return []

		best_actions = candidates[0][1]
		depth = 1

		try:
			while time.time() - start_time < time_budget:
				round_best_actions = None
				round_best_score = float('-inf')

				for i, actions in candidates:
					if time.time() - start_time >= time_budget:
						raise TimeoutError()
					simu_board, ok = board.forecast_turn(player_parity, actions)
					if not ok:
						continue

					score = self.mini_max(simu_board, player_parity, depth - 1, float('-inf'), float('inf'), True, start_time, time_budget)
					
					if score > round_best_score:
						round_best_score = score
						round_best_actions = actions

				if round_best_actions is not None:
					best_actions = round_best_actions
				depth += 1
				if depth > 8:
					break

		except TimeoutError:
			pass

		return best_actions

	def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
		"""
		Allows for you to display a string at the end of the match on our online
		portal for your own statistics usage. Be careful, your opponents will be 
		able to see this as well.
		"""
		# return ""
		p = board.get_player(player_parity)
		territory = board.get_territory_count(player_parity)
		return (
            f"stamina={p.stamina}/{p.max_stamina} "
            f"hills={len(p.controlled_hills)} "
            f"territory={territory}"
			f"time_left={time_left():.1f}s"
        )
	
	## minimax
	def mini_max(self, board: Board, player_parity: int, depth: int, alpha: float, beta: float, is_opp_turn: bool, start_time: float, time_budget: float) -> float:
		if time.time() - start_time >= time_budget:
			raise TimeoutError()
		
		winner = board.get_winner()
		if winner is not None:
			result, _ = winner
			if result == 0:
				return 0.0
			
			if result * player_parity > 0:
				return 1_000_000.0
			else:
				return -1_000_000.0
		
		if depth == 0:
			return self.evaluate_board(board, player_parity)
		
		if is_opp_turn:
			curr_parity = -player_parity
		else:
			curr_parity = player_parity
		
		## use manahattan candidates for inner nodes
		candidates = self._get_candidates_fast(board, curr_parity)
		if not candidates:
			return self.evaluate_board(board, player_parity)
		
		if not is_opp_turn:
			## maximizing
			best = float('-inf')
			for actions in candidates:
				simu_board, ok = board.forecast_turn(curr_parity, actions)
				if not ok:
					continue
				score = self.mini_max(simu_board, player_parity, depth - 1, alpha, beta, True, start_time, time_budget)
				
				if score > best:
					best = score
				alpha = max(alpha, score)
				if beta <= alpha:
					break
			if best > float('-inf'):
				return best
			else:
				return self.evaluate_board(board, player_parity)
		else:
			## minimzing
			best = float('inf')
			for actions in candidates:
				simu_board, ok = board.forecast_turn(curr_parity, actions)
				if not ok:
					continue
				score = self.mini_max(simu_board, player_parity, depth - 1, alpha, beta, False, start_time, time_budget)
				
				if score < best:
					best = score
				beta = min(beta, score)
				if beta <= alpha:
					break
			if best < float('inf'):
				return best
			else:
				return self.evaluate_board(board, player_parity)
	
	## candidate generation
	def _get_candidates(self, board: Board, player_parity: int) -> List[Tuple[float, List]]:
		## top level candidate generation - returns list of (score, [actions]) tuples
		player = board.get_player(player_parity)
		stamina = player.stamina
		results = []

		for d1 in Direction.cardinals():
			next_loc = player.loc + d1
			if board.oob(next_loc) or board.cells[next_loc.r][next_loc.c].is_wall:
				continue

			## greedy paint selection after move 1
			paint_budget = stamina - 25
			paints = self.choose_paints(board, player_parity, next_loc, paint_budget)
			actions = [Action.Move(d1)] + paints

			score = self._score_move_bfs(board, player_parity, next_loc)
			for p in paints:
				score += self.score_paint_target(board, player_parity, p.location) * 0.25
			results.append((score, actions))

			## second move option
			if stamina > GameConstants.EXTRA_MOVE_COST + 20:
				for d2 in Direction.cardinals():
					next_loc2 = next_loc + d2
                    
					if board.oob(next_loc2) or board.cells[next_loc2.r][next_loc2.c].is_wall:   
						continue
					score2 = score + 0.45 * self._score_move_bfs(board, player_parity, next_loc2)

					results.append((score2, [Action.Move(d1)] + paints + [Action.Move(d2)]))
		results.sort(key=lambda x: x[0], reverse=True)
		return results[:10]
	
	## fast candidate generation using manhattan distance
	def _get_candidates_fast(self, board: Board, player_parity: int) -> List[List]:
		player = board.get_player(player_parity)
		stamina = player.stamina
		results = []

		for d1 in Direction.cardinals():
			next_loc = player.loc + d1
            
			if board.oob(next_loc) or board.cells[next_loc.r][next_loc.c].is_wall:  
				continue
 
            
			paints = self.choose_paints(board, player_parity, next_loc, stamina - 25)
			actions = [Action.Move(d1)] + paints
			score = self._score_move_fast(board, player_parity, next_loc)
			results.append((score, actions))

			if stamina > GameConstants.EXTRA_MOVE_COST + 20:
				for d2 in Direction.cardinals():
					next_loc2 = next_loc + d2

					if board.oob(next_loc2) or board.cells[next_loc2.r][next_loc2.c].is_wall: 
						continue
                    
					score2 = score + 0.40 * self._score_move_fast(board, player_parity, next_loc2)
					results.append((score2, [Action.Move(d1)] + paints + [Action.Move(d2)]))
		results.sort(key=lambda x: x[0], reverse=True)
		return [i for _, i in results[:6]]

	## score a movement destination using manhattan distance
	def _score_move_fast(self, board: Board, player_parity: int, loc: Location) -> float:
		cell = board.cells[loc.r][loc.c]
		opponent_parity = -player_parity
		opponent = board.get_opponent(player_parity)
		score = 0.0
		player = board.get_player(player_parity)

		if loc == opponent.loc:
			if cell.owner_parity != opponent_parity:
				## we win collision yay
				return 1_000_000.0
			else:
				## we die nooooo :(
				return -500_000.0
		
		if cell.hill_id != 0:
			hill = board.hills[cell.hill_id]
			if hill.controller_parity == opponent_parity:
				## enemy hill - high priority
				score += 420
			elif hill.controller_parity == 0:
				## neutral hill
				score += 280
			else:
				## our hill - low priority
				score += 50
		
		## powerup - more valuable when stmaina is low
		if cell.powerup:
			stamina_ratio = player.stamina / max(1, player.max_stamina)
			score += 160 * (1.3 - stamina_ratio)
		
		## territory value
		if cell.owner_parity == 0:
			## expand to new territory
			score += 30
		elif cell.owner_parity == player_parity:
			## staying in our territory is low value
			score += 5
		else:
			## opponent territory weakens it
			score += 70
		
		## opponent pressure
		dist = self.manhattan(loc, opponent.loc)
		if player.stamina > opponent.stamina + 10:
			## stamina rich
			score += max(0, 5 - dist) * 25
		elif player.stamina < opponent.stamina - 10:
			## stamina poor
			score -= max(0, 4 - dist) * 30
		
		## painting frontier bonus
		for d in Direction.cardinals():
			n = loc + d
			if board.oob(n):
				continue
			nc = board.cells[n.r][n.c]

			if nc.is_wall or nc.beacon_parity != 0:  
				continue
			if nc.owner_parity == 0:  
				score += 8
			elif nc.owner_parity == -player_parity:    
				score += 15
		
		## manhattan toward key tarrgets
		for hill_id, hill in board.hills.items():
			if hill.controller_parity == player_parity:
				## defend
				priority = 0.8
			elif hill.controller_parity == opponent_parity:
				## attack
				priority = 3.2
			else:
				## claim neutral
				priority = 2.2
			for hloc in hill.cells:
				d = self.manhattan(loc, hloc)
				score += priority * (28 / (1 + d))
		return score
	
	## score movement destination with bfs
	def _score_move_bfs(self, board: Board, player_parity: int, loc: Location) -> float:
		cell = board.cells[loc.r][loc.c]
		opponent_parity = -player_parity
		opponent = board.get_opponent(player_parity)
		player = board.get_player(player_parity)
		score = 0.0

		## collision
		if loc == opponent.loc:
			if cell.owner_parity != opponent_parity:
				## we win
				return 1_000_000.0
			else:
				## we die
				return -500_000.0
		
		## hill value
		if cell.hill_id != 0:
			hill = board.hills[cell.hill_id]
			if hill.controller_parity == opponent_parity:
				## enemy hill - high priority
				score += 420
			elif hill.controller_parity == 0:
				## neutral hill
				score += 280
			else:
				## our hill - low priority
				score += 50
		## powerup
		if cell.powerup:
			stamina_ratio = player.stamina / max(1, player.max_stamina)
			score += 160 * (1.3 - stamina_ratio)
		
		## territory
		if cell.owner_parity == 0:
			## expand to new territory
			score += 30
		elif cell.owner_parity == player_parity:
			## staying in our territory is low value
			score += 5
		else:
			## opponent territory weakens it
			score += 70
		
		## pressure
		dist = self.manhattan(loc, opponent.loc)
		if player.stamina > opponent.stamina + 10:
			## stamina rich
			score += max(0, 5 - dist) * 25
		elif player.stamina < opponent.stamina - 10:
			## stamina poor
			score -= max(0, 4 - dist) * 30
		
		## frontier paintinf bonus
		for d in Direction.cardinals():
			n = loc + d
			if board.oob(n):
				continue
			nc = board.cells[n.r][n.c]

			if nc.is_wall or nc.beacon_parity != 0:  
				continue
			if nc.owner_parity == 0:  
				score += 8
			elif nc.owner_parity == -player_parity:    
				score += 15
			
		## bfs guidance toward key targets
		dist_map = self.get_dist_map(board, loc)
		for hill_id, hill in board.hills.items():
			if hill.controller_parity == player_parity:
				## defend
				priority = 0.8
			elif hill.controller_parity == opponent_parity:
				## attack
				priority = 3.2
			else:
				## claim neutral
				priority = 2.2
			for hloc in hill.cells:
				if hloc in dist_map:
					d = dist_map[hloc]
					score += priority * (38 / (1 + d))
		## powerup bfs guidance
		for r in range(board.board_size.r):
			for c in range(board.board_size.c):
				if board.cells[r][c].powerup:
					p_loc = Location(r, c)
					if p_loc in dist_map:
						score += 22 / (1 + dist_map[p_loc])
		return score
	
	## evaluation for minimax leaf nodes
	def evaluate_board(self, board: Board, player_parity: int) -> float:
		winner = board.get_winner()
		if winner is not None:
			result, _ = winner
			if result == 0:
				return 0.0
			
			if result * player_parity > 0:
				return 1_000_000.0
			else:
				return -1_000_000.0
		
		score = 0.0
		opponent_parity = -player_parity
		player = board.get_player(player_parity)
		opponent = board.get_opponent(player_parity)

		## hill control
		my_hills = len(player.controlled_hills)
		opp_hills = len(opponent.controlled_hills)
		score += (my_hills - opp_hills) * 3000

		## max stamina advantage
		score += (player.max_stamina - opponent.max_stamina) * 15

		## current stamina buffer
		score += (player.stamina - opponent.stamina) * 8

		## territory count
		player_territory = board.get_territory_count(player_parity)
		opp_territory = board.get_territory_count(opponent_parity)
		score += (player_territory - opp_territory) * 4

		## progress towards capturing hills
		for hill_id, hill in board.hills.items():
			if hill.controller_parity != player_parity:
				## contesting or trying to claim hill
				if player_parity > 0:
					player_cells = hill.control_positive
				else:
					player_cells = -hill.control_negative
				score += player_cells * 65
			else:
				## we control it
				if player_parity > 0:
					player_cells = hill.control_positive
				else:
					player_cells = -hill.control_negative
		
		## bfs proximity to uncontrolled hills
		dist_map = self.get_dist_map(board, player.loc)
		for hill_id, hill in board.hills.items():
			if hill.controller_parity == player_parity:
				continue
			if hill.controller_parity == opponent_parity:
				priority = 3.5
			else:
				priority = 2.5
			for hloc in hill.cells:
				if hloc in dist_map:
					d = dist_map[hloc]
					score += priority * 32 / (1 + dist_map[hloc])
		return score
	
	## return bfs distance from all loc to all reachable cells
	def get_dist_map(self, board: Board, loc: Location) -> Dict[Location, int]:
		if loc not in self._bfs_cache:
			self._bfs_cache[loc] = self.compute_dist_map(board, loc)
		return self._bfs_cache[loc]

	## compute the dist map
	def compute_dist_map(self, board: Board, start: Location) -> Dict[Location, int]:
		dist: Dict[Location, int] = {start: 0}
		q: deque = deque([start])

		while q:
			curr = q.popleft()
			for d in Direction.cardinals():
				next = curr + d
				if board.oob(next) or board.cells[next.r][next.c].is_wall or next in dist:
					continue
				dist[next] = dist[curr] + 1
				q.append(next)
		return dist

	## returns list of paint actions
	def choose_paints(self, board: Board, player_parity: int, from_loc: Location, budget: int) -> List[Action.Paint]:
		if budget < GameConstants.PAINT_STAMINA_COST:
			return []
		
		# scored: List[Tuple[float, Location]] = []
		scored = []

		for d in Direction.cardinals():
			target = from_loc + d
			if board.oob(target):
				continue
			cell = board.cells[target.r][target.c]
			if cell.is_wall or cell.beacon_parity != 0:
				continue
			# can only pain unowned or owned cells that arent opp owned
			if cell.owner_parity == -player_parity:
				continue

			score = self.score_paint_target(board, player_parity, target)
			if score > 5:
				scored.append((score, target))
		
		scored.sort(key=lambda x: x[0], reverse=True)

		paints: List[Action.Paint] = []
		remaining = budget
		for score, target in scored:
			if remaining >= GameConstants.PAINT_STAMINA_COST:
				paints.append(Action.Paint(target))
				remaining -= GameConstants.PAINT_STAMINA_COST
		return paints

	def score_paint_target(self, board: Board, player_parity: int, loc: Location) -> float:
		cell = board.cells[loc.r][loc.c]
		score = 0.0
		opponent_parity = -player_parity

		## new territory
		if cell.owner_parity == 0:
			score += 50
		elif cell.owner_parity == opponent_parity:
			layers = abs(cell.paint_value)
			score += max(0, GameConstants.MAX_PAINT_LAYERS - layers) * 8
		
		## hill bonus
		if cell.hill_id != 0:
			hill = board.hills[cell.hill_id]
			# if hill.controller_parity == player_parity:
			if hill.controller_parity == opponent_parity:
				## painting toward capturing a hill
				score += 160
			elif hill.controller_parity == 0:
				score += 110
			else:
				## reinforcing a hill we control
				score += 35
		
		return score

	@staticmethod
	def manhattan(a: Location, b: Location) -> int:
		return abs(a.r - b.r) + abs(a.c - b.c)
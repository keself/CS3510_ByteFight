from collections.abc import Callable, Iterable
from typing import Union, List, Optional, Tuple, Dict
from collections import deque

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
		self._bfs_cache: Optional[Dict[Tuple, Direction]] = None
		self._bfs_cache_turn: int = -1
		self._prev_loc: Optional[Location] = None
		self._stuck_count: int = 0
		
	
	def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
		"""
		Called at the start of each round. Return the number of stamina you
		want to bid for initiative. Defaults to zero.
		"""
		return 1
	
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
		# available = []
		# for dir in Direction:
		# 	next_loc = board.get_player(player_parity).loc + dir
		# 	if not board.oob(next_loc) and not board.cells[next_loc.r][next_loc.c].is_wall:
		# 		available.append(Action.Move(dir))

		# return random.choice(available)

		player = board.get_player(player_parity)
		my_loc = player.loc
		stamina = player.stamina
		opponent = board.get_opponent(player_parity)

		## detect getting stuck
		# if self._prev_loc == my_loc:
		# 	self._stuck_count += 1
		# else:
		# 	self._stuck_count = 0
		# self._prev_loc = my_loc
		
		actions: List[Union[Action.Move, Action.Paint]] = []
	
		## choose movement direction
		## mvoe 1
		move_dir = self.choose_direction(board, player_parity)
		if move_dir is None:
			return []  # completely walled in (shouldn't happen)
 		
		actions.append(Action.Move(move_dir))
		new_loc = my_loc + move_dir

		## aggressive painting
		buffer = max(10, int(stamina * 0.2))
		paint_budget = stamina - buffer

		paint_actions = self.choose_paints(board, player_parity, new_loc, paint_budget)
		actions.extend(paint_actions)

		## move 2
		if stamina > 50:
			bonus_dir = self.choose_direction(board, player_parity, from_loc = new_loc)

			if bonus_dir:
				bonus_loc = new_loc + bonus_dir
				bonus_cell = board.cells[bonus_loc.r][bonus_loc.c]

				## prioritize hills, powerups, or chasing opps
				if (bonus_cell.hill_id != 0 or bonus_cell.powerup or self.manhattan(bonus_loc, opponent.loc) <= 1):
					actions.append(Action.Move(bonus_dir))

		## use remaining stamina on painting with a buffer
		## paining costs 15 each
		## buffer: at min 20 stamina. if low, keep 30
		# if stamina > 60:
		# 	buffer = 20
		# else:
		# 	buffer = 30
		# paint_budget = stamina - buffer
		# paint_actions = self.choose_paints(board, player_parity, new_loc, paint_budget)
		# actions.extend(paint_actions)

		# ## second move if we have extra stamina and its valuable
		# ## only do it when rushing a hill or going after a win
		# if stamina - 10 >= buffer + 10 and len(paint_actions) == 0:
		# 	bonus_dir = self.choose_direction(board, player_parity, from_loc = new_loc, moves_done = 1)

		# 	if bonus_dir is not None:
		# 		bonus_loc = new_loc + bonus_dir
		# 		bonus_cell = board.cells[bonus_loc.r][bonus_loc.c]

		# 		## only worth it if target is a hill cell, powerup, or winning collision
		# 		opponent = board.get_opponent(player_parity)
		# 		if (bonus_cell.hill_id != 0 or bonus_cell.powerup or (bonus_loc == opponent.loc and bonus_cell.owner_parity != -player_parity)):
		# 			actions.append(Action.Move(bonus_dir))
		return actions
	
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
        )
	
	## movement logic
	def choose_direction(self, board: Board, player_parity: int, from_loc: Optional[Location] = None, moves_done: int = 0) -> Optional[Direction]:
		## pick the best cardinal direction
		player = board.get_player(player_parity)
		if from_loc is None:
			from_loc = player.loc
		
		opponent = board.get_opponent(player_parity)
		opponent_parity = -player_parity

		best_score = float('-inf')
		best_dir = None

		for d in Direction.cardinals():
			next_loc = from_loc + d
			if board.oob(next_loc):
				continue
			cell = board.cells[next_loc.r][next_loc.c]
			if cell.is_wall:
				continue

			## hard skip - collision we'd lose
			if next_loc == opponent.loc and cell.owner_parity == opponent_parity:
				continue
			
			score = self.score_move_target(board, player_parity, next_loc, from_loc)
			if score > best_score:
				best_score = score
				best_dir = d
			
		return best_dir
	
	def score_move_target(self, board: Board, player_parity: int, loc: Location, from_loc: Location) -> float:
		## score a pontential destination cell for movement
		cell = board.cells[loc.r][loc.c]
		opponent_parity = -player_parity
		opponent = board.get_opponent(player_parity)
		score = 0.0

		## instant win - winning collision
		if loc == opponent.loc:
			if cell.owner_parity != opponent_parity:
				return 1_000_000.0
		
		## hill priority
		if cell.hill_id != 0:
			hill = board.hills[cell.hill_id]
			if hill.controller_parity == opponent_parity:
				## enemy hill - high priority
				score += 300
			elif hill.controller_parity == player_parity:
				## unclaimed hill
				score += 200
			else:
				## our hill - low priority
				score += 40
		
		## powerup
		if cell.powerup:
			## more valuable when low on stamina
			player = board.get_player(player_parity)
			stamina_ratio = player.stamina / max(1, player.max_stamina)
			score += 60 + 80 * (1.0 - stamina_ratio)
		
		## territory valie
		if cell.owner_parity == 0:
			## expand to new territory
			score += 20
		elif cell.owner_parity == player_parity:
			## staying in our territory is low value
			score += 3
		else:
			## opponent territory weakens it
			score += 12
		
		## painting frontier
		paintable_score = 0.0
		for d2 in Direction.cardinals():
			n = loc + d2
			if not board.oob(n):
				nc = board.cells[n.r][n.c]
				if nc.is_wall or nc.beacon_parity != 0:
					continue
				if nc.owner_parity == 0:
					bonus = 8.0
					if nc.hill_id != 0:
						bonus += 40.0
					paintable_score += bonus
				elif nc.owner_parity == player_parity and nc.paint_value * player_parity < GameConstants.MAX_PAINT_VALUE:
					## can reinforce
					paintable_score += 1.0
		score += paintable_score

		## bfs guidance toward nearest priority target
		bfs_bonus = self.bfs_guidance_score(board, player_parity, loc)
		score += bfs_bonus

		## regen density bonus - we prefer cells near our own paint
		friendly_nearby = self.count_owned_nearby(board, player_parity, loc, radius = 2)
		score += friendly_nearby * 1.5

		## anti stuck - penalize revisiting last position
		if loc == self._prev_loc and self._stuck_count > 1:
			score -= 30 * self._stuck_count
		
		return score
	
	## bfs guidance

	## compute how well moving to candidate location goes w bfs shortest path to high val targets
	def bfs_guidance_score(self, board: Board, player_parity: int, candidate_loc: Location) -> float:
		## find highest priority target
		targets = self.get_priority_targets(board, player_parity)
		if not targets:
			return 0.0
		
		player = board.get_player(player_parity)

		## distance heuristic - reward moves that reduce distance to targets
		score = 0.0
		for priority, target_loc in targets:
			current_dist = self.manhattan(player.loc, target_loc)
			candidate_dist = self.manhattan(candidate_loc, target_loc)

			## reward getting close - extra reward for high priority targets
			delta = current_dist - candidate_dist
			score += delta * priority * 10
		return score
	
	def get_priority_targets(self, board: Board, player_parity: int) -> List[Tuple[float, Location]]:
		## return list of high priority target locations w priority score
		targets = []

		## hill cells we dont control
		for hill_id, hill in board.hills.items():
			if hill.controller_parity == player_parity:
				continue
			if hill.controller_parity == -player_parity:
				priority = 3.0
			else:
				priority = 2.0
			for hcell_loc in hill.cells:
				cell = board.cells[hcell_loc.r][hcell_loc.c]
				if cell.owner_parity != player_parity:
					targets.append((priority, hcell_loc))
		
		## powerups - lower priorty than hills
		for r in range(board.board_size.r):
			for c in range(board.board_size.c):
				if board.cells[r][c].powerup:
					targets.append((1.0, Location(r, c)))
		
		return targets
	
	## painting logic

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
			# if cell.beacon_parity != 0:
				# continue

			## can only pain unowned or owned cells that arent opp owned
			if cell.owner_parity == -player_parity:
				continue
			## skip if already at max layers
			if cell.owner_parity == player_parity and abs(cell.paint_value) >= GameConstants.MAX_PAINT_VALUE:
				continue

			score = self.score_paint_target(board, player_parity, target, cell)
			scored.append((score, target))
		
		scored.sort(key=lambda x: x[0], reverse=True)

		paints = []
		remaining = budget
		for _, target in scored:
			if remaining >= GameConstants.PAINT_STAMINA_COST:
				paints.append(Action.Paint(target))
				remaining -= GameConstants.PAINT_STAMINA_COST
			else:
				break
		return paints

	def score_paint_target(self, board: Board, player_parity: int, loc: Location, cell: CellState) -> float:
		score = 0.0
		opponent_parity = -player_parity

		## new territory
		if cell.owner_parity == 0:
			score += 40
		
		## hill bonus
		if cell.hill_id != 0:
			hill = board.hills[cell.hill_id]
			# if hill.controller_parity == player_parity:
			if hill.controller_parity == opponent_parity:
				## painting toward capturing a hill
				score += 150
			elif hill.controller_parity == 0:
				score += 100
			else:
				## reinforcing a hill we control
				score += 20
		
		## reinforce thin paint to make it harder to erase
		if cell.owner_parity == player_parity:
			## the thinner the paint, the higher the priority to reinforce
			layers = abs(cell.paint_value)
			score += max(0, 3 - layers) * 5
		
		return score

	def count_owned_nearby(self, board: Board, player_parity: int, loc: Location, radius: int) -> int:
		count = 0
		for dr in range(-radius, radius + 1):
			for dc in range(-radius, radius + 1):
				n = Location(loc.r + dr, loc.c + dc)
				if not board.oob(n) and board.cells[n.r][n.c].owner_parity == player_parity:
					count += 1
		return count

	@staticmethod
	def manhattan(a: Location, b: Location) -> int:
		return abs(a.r - b.r) + abs(a.c - b.c)
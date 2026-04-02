from collections.abc import Callable, Iterable
from typing import Union, List, Optional, Tuple, Dict
from collections import deque
import time

from game import *
import random


class PlayerController:
	"""
	You may add functions, however, __init__, bid, and play are the entry
	points for your program and should not be changed.
	"""
	
	## finite state machine states
	STATE_EXPAND = 0
	STATE_CAPTURE = 1
	STATE_SURVIVE = 2
	STATE_PRESSURE = 3

	def __init__(self, player_parity:int, time_left: Callable):
		self.player_parity = player_parity
		
		## bfs path - map target location -> next step direction
		self._bfs_cache: Dict[Location, Dict[Location, int]] = {}
		self._bfs_cache_turn: int = -1
		self.transposition_table: Dict = {}
		self._recent_positions: deque = deque(maxlen=8)

		## opponent tracking
		self._opp_history: deque = deque(maxlen=4)

		## fsm state
		self._state = self.STATE_EXPAND
		self._turn = 0

	def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
		"""
		Bid aggressively when a hill flip is imminent (ours or opponent's),
		and also when we're stamina-rich.
		"""
		player = board.get_player(player_parity)
		opponent = board.get_opponent(player_parity)

		## hill flip imminent - high priority bid
		for hill in board.hills.values():
			if self._cells_to_flip(hill, player_parity) <= 1:
				return 20
			if self._cells_to_flip(hill, -player_parity) <= 1:
				return 20

		## stamina advantage bid
		if player.stamina > opponent.stamina + 20:
			return 5
		return 3

	def play(
		self,
		board: Board,
		player_parity: int,
		time_left: Callable,
	) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:

		self._turn += 1

		## track opponent position for velocity prediction
		opp = board.get_opponent(player_parity)
		self._opp_history.append(opp.loc)

		## update fsm state
		self._state = self._compute_state(board, player_parity)

		## refresh bfs cache each turn
		if board.turn_count != self._bfs_cache_turn:
			self._bfs_cache.clear()
			self._bfs_cache_turn = board.turn_count
			self._recent_positions.append(board.get_player(player_parity).loc)

		## clear transposition table every so often
		if board.turn_count % 40 == 0:
			self.transposition_table.clear()

		## SURVIVE state: retreat to friendly territory and regen
		if self._state == self.STATE_SURVIVE:
			return self._survive_turn(board, player_parity)

		## time management
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
		states = ["EXPAND", "CAPTURE", "SURVIVE", "PRESSURE"]
		p = board.get_player(player_parity)
		territory = board.get_territory_count(player_parity)
		return (
			f"state={states[self._state]} "
			f"stamina={p.stamina}/{p.max_stamina} "
			f"hills={len(p.controlled_hills)} "
			f"territory={territory} "
			f"time_left={time_left():.1f}s"
		)

	def _compute_state(self, board: Board, parity: int) -> int:
		player = board.get_player(parity)
		opp_parity = -parity

		## low stamina - survive and regen
		if player.stamina < 8:
			return self.STATE_SURVIVE

		## hill flip imminent for either side - capture mode
		for hill in board.hills.values():
			if self._cells_to_flip(hill, parity) <= 2:
				return self.STATE_CAPTURE
			if self._cells_to_flip(hill, opp_parity) <= 2:
				return self.STATE_CAPTURE

		## opponent controls more hills than us - go capture
		my_controlled = sum(1 for h in board.hills.values() if h.controller_parity == parity)
		opp_controlled = sum(1 for h in board.hills.values() if h.controller_parity == opp_parity)
		if opp_controlled > my_controlled:
			return self.STATE_CAPTURE

		if self._turn < 50:
			return self.STATE_EXPAND

		return self.STATE_PRESSURE
	
	## when stamina is super love, move to safety and regen
	def _survive_turn(self, board: Board, player_parity: int) -> list:
		player = board.get_player(player_parity)
		best_dir = None
		best_score = float('-inf')

		for direction in Direction.cardinals():
			next_loc = player.loc + direction
			if board.oob(next_loc):
				continue
			cell = board.cells[next_loc.r][next_loc.c]
			if cell.is_wall:
				continue

			score = 0.0

			## strongly prefer our own territory for regen
			if cell.owner_parity == player_parity:
				score += 80
			elif cell.owner_parity == 0:
				score += 20
			else:
				score -= 40

			## avoid opponent when weak
			opp = board.get_opponent(player_parity)
			dist = self.manhattan(next_loc, opp.loc)
			score += dist * 15

			## antioscillation
			score -= self._repeat_position_penalty(next_loc)

			if score > best_score:
				best_score = score
				best_dir = direction

		if best_dir is None:
			return []

		actions = [Action.Move(direction=best_dir)]
		## still paint if we have any budget left
		new_loc = player.loc + best_dir
		paints = self.choose_paints(board, player_parity, new_loc, player.stamina)
		actions.extend(paints)
		return actions

	## opponent velocity from last 2 pos
	def _opp_velocity(self) -> Optional[Tuple[int, int]]:
		if len(self._opp_history) < 2:
			return None
		a = self._opp_history[-2]
		b = self._opp_history[-1]
		dr, dc = b.r - a.r, b.c - a.c
		if abs(dr) + abs(dc) == 1:
			return (dr, dc)
		return None

	## return predicted next opp loc from vel
	def _predicted_opp_loc(self, board: Board) -> Optional[Location]:
		opp = board.get_opponent(self.player_parity)
		vel = self._opp_velocity()
		if vel is None:
			return None
		pred = Location(opp.loc.r + vel[0], opp.loc.c + vel[1])
		if board.oob(pred) or board.cells[pred.r][pred.c].is_wall:
			return None
		return pred

	## deferred collisions afety
	def _move_is_safe(self, board: Board, parity: int, loc: Location) -> bool:
		opp_parity = -parity
		opp = board.get_opponent(parity)
		cell = board.cells[loc.r][loc.c]

		## immediate loss: stepping onto opponent on their painted cell
		if loc == opp.loc and cell.owner_parity == opp_parity:
			return False

		## deferred loss: opponent is adjacent and cell is not our paint
		opp_dist = abs(opp.loc.r - loc.r) + abs(opp.loc.c - loc.c)
		if opp_dist <= 1 and cell.owner_parity != parity:
			return False

		return True

	## only block immediat elosing collisions
	def _move_is_safe_relaxed(self, board: Board, parity: int, loc: Location) -> bool:
		opp_parity = -parity
		opp = board.get_opponent(parity)
		cell = board.cells[loc.r][loc.c]
		if loc == opp.loc and cell.owner_parity == opp_parity:
			return False
		return True

	## place beacon is window density met and we're near a hill we control
	def _should_place_beacon(self, board: Board, parity: int, pos: Location) -> bool:
		if board.oob(pos):
			return False
		player = board.get_player(parity)
		cell = board.cells[pos.r][pos.c]
		opp_parity = -parity

		if cell.owner_parity == opp_parity:
			return False
		if cell.beacon_parity != 0:
			return False
		if player.beacon_count >= 3:
			return False

		## check window density requirement
		window_radius = GameConstants.BEACON_WINDOW_SIZE_P // 2
		friendly_count, valid_count = 0, 0
		for dr in range(-window_radius, window_radius + 1):
			for dc in range(-window_radius, window_radius + 1):
				loc = Location(pos.r + dr, pos.c + dc)
				if board.oob(loc):
					continue
				c = board.cells[loc.r][loc.c]
				if c.is_wall:
					continue
				valid_count += 1
				if c.owner_parity == parity:
					friendly_count += 1

		if valid_count == 0:
			return False
		if (friendly_count * (GameConstants.BEACON_WINDOW_SIZE_P ** 2)
				< GameConstants.BEACON_REQUIREMENT_Q * valid_count):
			return False

		## strategic value - on or adjacent to a hill we control
		if cell.hill_id != 0:
			hill = board.hills[cell.hill_id]
			if hill.controller_parity == parity:
				return True

		for d in Direction.cardinals():
			adj = pos + d
			if board.oob(adj):
				continue
			adj_cell = board.cells[adj.r][adj.c]
			if adj_cell.hill_id != 0:
				hill = board.hills[adj_cell.hill_id]
				if hill.controller_parity == parity:
					return True

		## place first beacon anywhere requirements are met
		if player.beacon_count == 0:
			return True

		return False

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
		
		curr_parity = -player_parity if is_opp_turn else player_parity
		candidates = self._get_candidates_fast(board, curr_parity)
		if not candidates:
			return self.evaluate_board(board, player_parity)
		
		if not is_opp_turn:
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
			return best if best > float('-inf') else self.evaluate_board(board, player_parity)
		else:
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
			return best if best < float('inf') else self.evaluate_board(board, player_parity)

	## candidate generation
	def _get_candidates(self, board: Board, player_parity: int) -> List[Tuple[float, List]]:
		player = board.get_player(player_parity)
		stamina = player.stamina
		results = []
		objective = self._choose_objective(board, player_parity, player.loc)

		for move1, next_loc, base_score, move_cost in self._iter_move_options(board, player_parity, player.loc, moves_taken=0):
			paint_budget = stamina - move_cost
			paints = self.choose_paints(board, player_parity, next_loc, paint_budget)

			## try placing a beacon on arrival - beacon costs no stamina so paint budget unchanged
			place_beacon = self._should_place_beacon(board, player_parity, next_loc)
			move1_final = Action.Move(direction=move1.direction, move_type=move1.move_type, place_beacon=place_beacon)
			actions = [move1_final] + paints

			score = base_score
			for p in paints:
				score += self.score_paint_target(board, player_parity, p.location) * 0.35
			score += self._objective_progress(board, player.loc, next_loc, objective)
			score -= self._repeat_position_penalty(next_loc)
			results.append((score, actions))

			second_move_cost = GameConstants.EXTRA_MOVE_COST
			if stamina >= move_cost + second_move_cost:
				for move2, next_loc2, step_score2, move2_cost in self._iter_move_options(
					board, player_parity, next_loc, moves_taken=1,
				):
					if next_loc2 == player.loc:
						continue
					if move_cost + move2_cost > stamina:
						continue
					remaining_after_moves = stamina - move_cost - move2_cost
					paints2 = self.choose_paints(board, player_parity, next_loc2, remaining_after_moves)
					score2 = base_score + 0.45 * step_score2
					for p in paints2:
						score2 += self.score_paint_target(board, player_parity, p.location) * 0.35
					score2 += self._objective_progress(board, player.loc, next_loc2, objective) * 1.35
					score2 -= self._repeat_position_penalty(next_loc2)
					results.append((score2, [move1, move2] + paints2))

		results.sort(key=lambda x: x[0], reverse=True)
		return results[:14]

	def _get_candidates_fast(self, board: Board, player_parity: int) -> List[List]:
		player = board.get_player(player_parity)
		stamina = player.stamina
		results = []
		objective = self._choose_objective(board, player_parity, player.loc)

		for move1, next_loc, base_score, move_cost in self._iter_move_options(board, player_parity, player.loc, moves_taken=0):
			paints = self.choose_paints(board, player_parity, next_loc, stamina - move_cost)
			actions = [move1] + paints
			score = base_score
			score += self._objective_progress(board, player.loc, next_loc, objective)
			score -= self._repeat_position_penalty(next_loc)
			results.append((score, actions))

			if stamina >= move_cost + GameConstants.EXTRA_MOVE_COST:
				for move2, next_loc2, step_score2, move2_cost in self._iter_move_options(
					board, player_parity, next_loc, moves_taken=1,
				):
					if next_loc2 == player.loc:
						continue
					if move_cost + move2_cost > stamina:
						continue
					remaining_after_moves = stamina - move_cost - move2_cost
					paints2 = self.choose_paints(board, player_parity, next_loc2, remaining_after_moves)
					score2 = base_score + 0.40 * step_score2
					score2 += self._objective_progress(board, player.loc, next_loc2, objective) * 1.35
					score2 -= self._repeat_position_penalty(next_loc2)
					results.append((score2, [move1, move2] + paints2))

		results.sort(key=lambda x: x[0], reverse=True)
		return [i for _, i in results[:8]]

	def _iter_move_options(self, board, player_parity, from_loc, moves_taken):
		opponent = board.get_opponent(player_parity)
		player = board.get_player(player_parity)
		base_move_cost = 0 if moves_taken == 0 else GameConstants.EXTRA_MOVE_COST * moves_taken

		for direction in Direction.cardinals():
			next_loc = from_loc + direction
			if board.oob(next_loc):
				continue
			cell = board.cells[next_loc.r][next_loc.c]
			if cell.is_wall:
				continue

			regular_score = self._score_move(board, player_parity, next_loc, use_erase=False)
			yield Action.Move(direction=direction), next_loc, regular_score, base_move_cost

			if player.stamina < base_move_cost + GameConstants.ERASE_STEP_EXTRA_COST:
				continue
			if next_loc == opponent.loc:
				continue
			if cell.owner_parity != -player_parity:
				continue

			erase_score = self._score_move(board, player_parity, next_loc, use_erase=True)
			yield (
				Action.Move(direction=direction, move_type=MoveType.ERASE),
				next_loc,
				erase_score,
				base_move_cost + GameConstants.ERASE_STEP_EXTRA_COST,
			)

	## move scoring
	def _score_move(self, board, player_parity, loc, use_erase=False, use_bfs=False):
		cell = board.cells[loc.r][loc.c]
		opponent_parity = -player_parity
		opponent = board.get_opponent(player_parity)
		score = 0.0
		player = board.get_player(player_parity)

		if loc == opponent.loc:
			if cell.owner_parity != opponent_parity:
				return 1_000_000.0
			else:
				return -500_000.0

		dist = self.manhattan(loc, opponent.loc)

		## collision safety gate - adjacent to opponent on non-our-paint is dangerous
		if dist <= 1 and cell.owner_parity != player_parity:
			stamina_deficit = opponent.stamina - player.stamina
			base_penalty = 300 + max(0, stamina_deficit) * 4
			score -= base_penalty

		## 2-step lookahead - opponent can reach this cell next turn
		## penalise landing here if it's not our paint and opponent is 2 steps away
		elif dist == 2 and cell.owner_parity != player_parity:
			stamina_deficit = opponent.stamina - player.stamina
			score -= 80 + max(0, stamina_deficit) * 2

		if cell.hill_id != 0:
			hill = board.hills[cell.hill_id]
			## only give full hill bonus if it's safe to be here
			hill_safe = (dist > 1 or cell.owner_parity == player_parity)
			## early game multiplier - hills matter most in first 40 turns
			early_mult = max(1.0, 2.5 - self._turn * 0.04)
			if hill.controller_parity == opponent_parity:
				score += (420 if hill_safe else 80) * early_mult
			elif hill.controller_parity == 0:
				score += (280 if hill_safe else 60) * early_mult
			else:
				score += 50

		if cell.powerup:
			stamina_ratio = player.stamina / max(1, player.max_stamina)
			score += 160 * (1.3 - stamina_ratio)

		if cell.owner_parity == 0:
			score += 30
		elif cell.owner_parity == player_parity:
			score += 5
		else:
			score += 30 if use_erase else 70
			if use_erase:
				score += 60 + min(40, abs(cell.paint_value) * 12)

		## opponent pressure when safe
		if dist > 1:
			if player.stamina > opponent.stamina + 10:
				score += max(0, 5 - dist) * 25
			elif player.stamina < opponent.stamina - 10:
				score -= max(0, 4 - dist) * 30

		if dist == 1 and player.stamina >= opponent.stamina + 15 and cell.owner_parity == player_parity:
			score += 65
		elif dist == 1 and player.stamina + 15 < opponent.stamina:
			score -= 80

		## painting frontier bonus
		for d in Direction.cardinals():
			n = loc + d
			if board.oob(n):
				continue
			nc = board.cells[n.r][n.c]
			if nc.is_wall or nc.beacon_parity == player_parity:
				continue
			if nc.owner_parity == 0:
				score += 8
			elif nc.owner_parity == -player_parity:
				score += 15
			if nc.beacon_parity == opponent_parity:
				score += 45

		## opponent velocity - bonus for intercepting predicted path
		pred_opp = self._predicted_opp_loc(board)
		if pred_opp is not None and loc == pred_opp:
			if cell.owner_parity == player_parity:
				score += 80
			else:
				score += 30

		if use_bfs:
			dist_map = self.get_dist_map(board, loc)
		else:
			dist_map = None

		## guidance toward key targets
		for hill_id, hill in board.hills.items():
			if hill.controller_parity == player_parity:
				priority = 0.8
			elif hill.controller_parity == opponent_parity:
				priority = 3.2
			else:
				priority = 2.2
			for hloc in hill.cells:
				if use_bfs:
					if hloc in dist_map:
						score += priority * (38 / (1 + dist_map[hloc]))
				else:
					d = self.manhattan(loc, hloc)
					score += priority * (28 / (1 + d))

		if use_bfs:
			for r in range(board.board_size.r):
				for c in range(board.board_size.c):
					if board.cells[r][c].powerup:
						p_loc = Location(r, c)
						if p_loc in dist_map:
							score += 22 / (1 + dist_map[p_loc])

		score += self._frontier_pressure(board, loc, player_parity, dist_map if use_bfs else None)
		score += 4 * self._count_local_control(board, loc, player_parity)
		return score

	## board evalutation

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

		## hill-flip urgency-weight progress toward flipping
		for hill_id, hill in board.hills.items():
			flip_us = self._cells_to_flip(hill, player_parity)
			flip_opp = self._cells_to_flip(hill, opponent_parity)
			score += max(0, 5 - flip_us) * 120
			score -= max(0, 5 - flip_opp) * 120

		## max stamina advantage
		score += (player.max_stamina - opponent.max_stamina) * 40

		## current stamina buffer
		score += (player.stamina - opponent.stamina) * 8

		## territory count
		player_territory = board.get_territory_count(player_parity)
		opp_territory = board.get_territory_count(opponent_parity)
		score += (player_territory - opp_territory) * 4

		## local control
		score += (self._count_local_control(board, player.loc, player_parity) - self._count_local_control(board, opponent.loc, opponent_parity)) * 18

		## progress towards capturing hills
		for hill_id, hill in board.hills.items():
			if hill.controller_parity != player_parity:
				if player_parity > 0:
					player_cells = hill.control_positive
				else:
					player_cells = -hill.control_negative
				score += player_cells * 65
			else:
				if player_parity > 0:
					player_cells = hill.control_positive
				else:
					player_cells = -hill.control_negative
				score += player_cells * 18

		## bfs proximity to uncontrolled hills
		dist_map = self.get_dist_map(board, player.loc)
		for hill_id, hill in board.hills.items():
			if hill.controller_parity == player_parity:
				continue
			priority = 3.5 if hill.controller_parity == opponent_parity else 2.5
			for hloc in hill.cells:
				if hloc in dist_map:
					score += priority * 32 / (1 + dist_map[hloc])

		## beacon value
		for r in range(board.board_size.r):
			for c in range(board.board_size.c):
				cell = board.cells[r][c]
				if cell.beacon_parity == opponent_parity:
					score -= 140 / (1 + self.manhattan(player.loc, Location(r, c)))
				elif cell.beacon_parity == player_parity:
					score += 60 / (1 + self.manhattan(player.loc, Location(r, c)))

		## deferred collision danger - heavily penalise if opponent adjacent and not on our paint
		our_cell = board.cells[player.loc.r][player.loc.c]
		if our_cell.owner_parity != player_parity:
			dist = self.manhattan(player.loc, opponent.loc)
			if dist <= 1:
				stamina_deficit = opponent.stamina - player.stamina
				score -= 300 + max(0, stamina_deficit) * 4

		return score

	##painting stuff
	def choose_paints(self, board: Board, player_parity: int, from_loc: Location, budget: int) -> List[Action.Paint]:
		if budget < GameConstants.PAINT_STAMINA_COST:
			return []

		scored = []
		opp = board.get_opponent(player_parity)
		pred_opp = self._predicted_opp_loc(board)

		for d in Direction.cardinals():
			target = from_loc + d
			if board.oob(target):
				continue
			cell = board.cells[target.r][target.c]
			if cell.is_wall or cell.beacon_parity == player_parity:
				continue
			if cell.owner_parity == -player_parity:
				continue

			score = self.score_paint_target(board, player_parity, target)

			## velocity trap - paint the cell the opponent is predicted to walk into
			if pred_opp is not None and target == pred_opp:
				score += 90

			## also boost cells adjacent to opponent
			if self.manhattan(target, opp.loc) == 1:
				score += 30

			if score > 5:
				scored.append((score, target))

		scored.sort(key=lambda x: x[0], reverse=True)

		paints = []
		remaining = budget
		max_paints = min(3, budget // GameConstants.PAINT_STAMINA_COST)
		for score, target in scored:
			if remaining >= GameConstants.PAINT_STAMINA_COST and len(paints) < max_paints:
				paints.append(Action.Paint(target))
				remaining -= GameConstants.PAINT_STAMINA_COST
		return paints

	def score_paint_target(self, board: Board, player_parity: int, loc: Location) -> float:
		cell = board.cells[loc.r][loc.c]
		score = 0.0
		opponent_parity = -player_parity

		if cell.owner_parity == 0:
			score += 50
		elif cell.owner_parity == player_parity:
			score += max(0, GameConstants.MAX_PAINT_VALUE - abs(cell.paint_value)) * 6

		if cell.hill_id != 0:
			hill = board.hills[cell.hill_id]
			if hill.controller_parity == opponent_parity:
				score += 160
			elif hill.controller_parity == 0:
				score += 110
			else:
				score += 35

		if cell.beacon_parity == opponent_parity:
			score += 120

		if self._adjacent_to_opponent(board, loc, player_parity):
			score += 16

		return score

	## helpers

	## how many more cells need to be paint to flipp hill
	def _cells_to_flip(self, hill, parity: int) -> int:
		import math
		required = math.ceil(len(hill.cells) * GameConstants.HILL_CONTROL_THRESHOLD)
		current = hill.control_positive if parity > 0 else -hill.control_negative
		return max(0, required - current)

	def _count_local_control(self, board: Board, center: Location, player_parity: int) -> int:
		count = 0
		for dr in range(-2, 3):
			for dc in range(-2, 3):
				loc = Location(center.r + dr, center.c + dc)
				if board.oob(loc):
					continue
				if board.cells[loc.r][loc.c].owner_parity == player_parity:
					count += 1
		return count

	def _adjacent_to_opponent(self, board: Board, loc: Location, player_parity: int) -> bool:
		opponent = board.get_opponent(player_parity)
		return self.manhattan(loc, opponent.loc) == 1

	def _repeat_position_penalty(self, loc: Location) -> float:
		penalty = 0.0
		for idx, old_loc in enumerate(reversed(self._recent_positions)):
			if loc == old_loc:
				penalty += 18 + idx * 7
		return penalty

	def _frontier_pressure(self, board, loc, player_parity, dist_map=None):
		best = float('-inf')
		for r in range(board.board_size.r):
			for c in range(board.board_size.c):
				cell = board.cells[r][c]
				target = Location(r, c)
				value = 0.0

				if cell.is_wall:
					continue
				if cell.owner_parity != player_parity:
					value = 90.0 if cell.owner_parity == 0 else 130.0
				elif cell.hill_id != 0 and board.hills[cell.hill_id].controller_parity != player_parity:
					value = 150.0
				elif cell.powerup:
					value = 85.0
				elif cell.beacon_parity == -player_parity:
					value = 140.0
				else:
					continue

				if dist_map is not None:
					if target not in dist_map:
						continue
					d = dist_map[target]
				else:
					d = self.manhattan(loc, target)
				best = max(best, value / (1 + d))

		return 0.0 if best == float('-inf') else best

	def _choose_objective(self, board: Board, player_parity: int, start_loc: Location) -> Optional[Location]:
		dist_map = self.get_dist_map(board, start_loc)
		best_score = float('-inf')
		best_loc = None
		early_game = self._turn <= 40

		for r in range(board.board_size.r):
			for c in range(board.board_size.c):
				cell = board.cells[r][c]
				loc = Location(r, c)
				if cell.is_wall or loc not in dist_map:
					continue

				value = 0.0
				if cell.hill_id != 0:
					hill = board.hills[cell.hill_id]
					if hill.controller_parity == -player_parity:
						flip_bonus = max(0, 4 - self._cells_to_flip(hill, player_parity)) * 30
						value = 320.0 + flip_bonus
					elif hill.controller_parity == 0:
						flip_bonus = max(0, 4 - self._cells_to_flip(hill, player_parity)) * 20
						value = 280.0 + flip_bonus
					else:
						value = 40.0
				elif early_game:
					## in early game, only target hills — ignore everything else
					continue
				elif cell.beacon_parity == -player_parity:
					value = 200.0
				elif cell.powerup:
					value = 160.0
				elif cell.owner_parity == -player_parity:
					value = 130.0
				elif cell.owner_parity == 0:
					value = 100.0

				if value <= 0:
					continue

				d = dist_map[loc]
				score = value - 6 * d
				if score > best_score:
					best_score = score
					best_loc = loc

		return best_loc

	def _objective_progress(self, board, start_loc, end_loc, objective):
		if objective is None:
			return 0.0
		start_dist_map = self.get_dist_map(board, start_loc)
		if objective not in start_dist_map:
			return 0.0
		end_dist_map = self.get_dist_map(board, end_loc)
		if objective not in end_dist_map:
			return 0.0
		progress = start_dist_map[objective] - end_dist_map[objective]
		## weight progress more heavily early game so minimax rushes to hills
		early_mult = max(1.0, 2.5 - self._turn * 0.04)
		return progress * 26.0 * early_mult

	def get_dist_map(self, board: Board, loc: Location) -> Dict[Location, int]:
		if loc not in self._bfs_cache:
			self._bfs_cache[loc] = self.compute_dist_map(board, loc)
		return self._bfs_cache[loc]

	def compute_dist_map(self, board: Board, start: Location) -> Dict[Location, int]:
		dist: Dict[Location, int] = {start: 0}
		q: deque = deque([start])
		while q:
			curr = q.popleft()
			for d in Direction.cardinals():
				nxt = curr + d
				if board.oob(nxt) or board.cells[nxt.r][nxt.c].is_wall or nxt in dist:
					continue
				dist[nxt] = dist[curr] + 1
				q.append(nxt)
		return dist

	@staticmethod
	def manhattan(a: Location, b: Location) -> int:
		return abs(a.r - b.r) + abs(a.c - b.c)
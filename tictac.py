from engine import Engine
from players import Human, Random, Network
import random


def play_game(size, rand_start, print_mode = 1):
	e = Engine(size, print_mode)
	p1 = Network(1, size*size, e, print_mode)
	p2 = Random(2, size*size, e, print_mode)
	
	player_arr = [p1, p2]
	if rand_start == True:
		player_index = random.randint(0, 1)
	else:
		player_index = 0

	print(player_index, e.get_board())
	exit()
	while True:
		curr = player_arr[player_index]

		e.print_board()
		moves = e.get_legal_moves()
		t = e.is_terminal(moves)
		if t != -2:
			return t

		curr.print_turn()
		m = curr.get_move(moves)
		e.move(m, curr.get_sym())

		player_index = 1 - player_index


def main():
	draw = 0
	p1 = 0
	p2 = 0
	for i in range(50):
		winner = play_game(3, True, 1)
		if winner == 0:
			print("DRAW")
			draw += 1
		elif winner == 1:
			p1 += 1
			print("PLAYER {0} WINS".format(str(winner)))
		else:
			p2 += 1
			print("PLAYER {0} WINS".format(str(winner)))
		print('total {0}, p1 {1}, p2 {2}, draw {3}'.format(draw + p1 + p2, p1, p2, draw))
main()

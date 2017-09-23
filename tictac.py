from engine import Engine
from players import Human, Random, Network


def play_game(size, print_mode = 1):
	e = Engine(size, print_mode)
	p1 = Network(1, size*size, e, print_mode)
	p2 = Random(2, size*size, e, print_mode)
	curr = p1

	while True:
		e.print_board()
		moves = e.get_legal_moves()
		t = e.is_terminal(moves)
		if t != -2:
			return t

		curr.print_turn()
		m = curr.get_move(moves)
		e.move(m, curr.get_sym())
		if curr == p1:
			curr = p2
		else:
			curr = p1


def main():
	draw = 0
	p1 = 0
	p2 = 0
	for i in range(50):
		winner = play_game(3, 0)
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

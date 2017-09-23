from engine import Engine
from players import Human, Random
import h5py
import numpy as np
import os


def play_game(size, print_mode):
    e = Engine(size, print_mode)
    p1 = Random(1, size*size, e, print_mode)
    p2 = Random(2, size*size, e, print_mode)
    curr = p1
    curr_state = 0
    total_squares = size*size

    move_len = 0

    # Storing
    # 1 bit for symbol
    # 1 bit for ?????
    # 9 bits for player 1
    # 9 bits for player 2
    # 9 bits for move made 


    board_states = np.zeros(shape=(total_squares, 3+(total_squares*3)), dtype=np.int8)

    while True:
        e.print_board()
        moves = e.get_legal_moves()
        t = e.is_terminal(moves)
        if t != -2:
            return (t, board_states, move_len)

        curr.print_turn()
        m = curr.get_move(moves)

        # BOARD COPYING 
        
        b = e.get_board()

        board_states[curr_state][0] = curr.get_sym()
        board_states[curr_state][1] = 10

        for i in range(total_squares):
            if b[i] == 1:
                board_states[curr_state][i+3] = 1
            elif b[i] == 2:
                board_states[curr_state][i+12] = 1
        board_states[curr_state][21+m] = 1
        curr_state += 1
        
        # END BOARD COPYING 

        e.move(m, curr.get_sym())

        # SWAP

        move_len += 1
        # print(move_len)
        if curr == p1:
            curr = p2
        else:
            curr = p1



# STATE WRITE IN THIS FORMAT

# 1. 1
# 2. 1
# 3. 9
# 4. [0 1 0 0 0 0 1 0 0]
# 5. [0 0 0 0 0 0 0 0 1]
# 6. [0 0 0 0 0 1 0 0 0]
# 7. [0, 1, 0]
# 8. [0, 0, 0]
# 9. [1, 0, 2]


# 1: move color (1 or 2)        [indexes: 0]
# 2: winner (1 or 2)            [indexes: 1]
# 3: Length of game (5 - 9)     [indexes: 2]
# 4: player 1's pieces          [indexes: 3-11]
# 5: player 2's pieces          [indexes: 12-20]
# 6: move made (1 hot)          [indexes: 21-29]
# 7-9: board state              [made up from 4 and 5]

# 

def main():
    size = 3
    total_squares = size*size
    

    # path stuff
    my_path = os.getcwd()
    gen_type = 'random'
    data_path = os.path.join(my_path, 'data', gen_type, str(size))
    curr_files = os.listdir(data_path)
    amount = len([x for x in curr_files if not x.startswith('.')])
    path = os.path.join(data_path, 'mytestfile' + str(amount) + '.hdf5')
    
    i = 0
    # game_num = 0

    # BASED ON MOVES, WILL BE APPROXIMATE
    moves = 1000000
    with h5py.File(path, 'w') as f:
        dset = f.create_dataset('random_games', shape=(moves, 3+(total_squares*3)), dtype=np.int8)

        while i < moves:
            winner, game, move_len = play_game(size, 0)
            if winner == 0:
                # draw
                continue

            trunced_game = game[0:move_len]
            game[0][2] = move_len

            if move_len + i > moves:
                print('off', moves-i)
                break

            for state in trunced_game:
                state[1] = winner
                dset[i] = state
                # print(dset[i], i, move_len)
                i += 1


    # BASED ON GAMES
    # games = 5000000
    # with h5py.File(path, 'w') as f:
    #     dset = f.create_dataset('random_games', shape=(games, total_squares, 3+(total_squares*3)), dtype=np.int8)

    #     while i < games:
    #         winner, game, move_len = play_game(size, 0)

    #         if winner == 0:
    #             # DRAW
    #             continue

    #         game[0][2] = move_len

    #         for state in game:
    #             # if state[0] == 0:
    #             #     break
    #             state[1] = winner

    #         dset[i] = game
    #         i += 1
main()


from engine import Engine
from players import Human, Random, Network
import h5py
import numpy as np
import os
import random


def play_game(size, rand_start, start_state, print_mode):
    e = Engine(size, print_mode, start_state)
    p1 = Network(1, size*size, e, print_mode, 'policy_net_v2')
    p2 = Network(2, size*size, e, print_mode, 'policy_net_v2', 'all')
    curr = p1
    curr_state = 0
    total_squares = size*size

    move_len = 0

    player_arr = [p1, p2]
    if rand_start == True:
        player_index = random.randint(0, 1)
    else:
        player_index = 0

    # Storing
    # 1 bit for symbol
    # 1 bit for ?????
    # 9 bits for player 1
    # 9 bits for player 2
    # 9 bits for move made 

    if print_mode == 1:
        p1.announce_title()
        p2.announce_title()

    # print(player_index, e.get_board())
    # exit()

    board_states = np.zeros(shape=(total_squares, 3+(total_squares*3)), dtype=np.int8)

    while True:
        curr = player_arr[player_index]

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
        
        player_index = 1 - player_index



def spread_to_combined(b1, b2):
    arr = []
    for index in range(len(b1)):
        if b1[index] == 1:
            arr.append(1)
        elif b2[index] == 1:
            arr.append(2)
        else:
            arr.append(0)
    return arr


def random_state_gen(f, random_sample):
    for move_id in random_sample:
        state = f['random_games'][move_id]
        winner = 1
        if state[1] == 2:
            winner = 2

        move_color = state[0]
        game_len = state[2]

        p1_board = state[3:12]
        p2_board = state[12:21]
        yield spread_to_combined(p1_board, p2_board)
    yield None



def get_last(f, def_shape):
    def_shape -= 1
    if f['random_games'][def_shape][0] != 0:
        return def_shape
    while def_shape >= 0:
        if f['random_games'][def_shape][0] != 0:
            return def_shape + 1
        def_shape -= 1


def gen_start_states():
    size = 3
    path = os.path.join('data', 'random', str(size), 'mytestfile0.hdf5')

    with h5py.File(path, 'r') as f:
        moves_len = f['random_games'].shape[0]
        last_move_index = get_last(f, moves_len)

        games_range_gen = range(last_move_index+1)
        random_sample = random.sample(games_range_gen, last_move_index)

        print('training from {0} games'.format(last_move_index))

        state_gen = random_state_gen(f, random_sample)
        while True:
            board_state = next(state_gen)

            if board_state == 'over':
                print('finished {0} states, this batch only has {1}/{2} states'.format(last_move_index+1, i, batch_size))
                state_gen = random_state_gen(f, random_sample)
                continue
            yield board_state




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
    

    base_model_path = os.path.join('models', 'policy_net_v2')

    models = os.listdir(base_model_path)
    last_iter = len(models) - 1

    # path stuff
    my_path = os.getcwd()
    gen_type = 'neural'
    data_path = os.path.join(my_path, 'data', gen_type, str(size))
    curr_files = os.listdir(data_path)
    amount = len([x for x in curr_files if not x.startswith('.')])
    path = os.path.join(data_path, str(amount) + '_model_iter_' + str(last_iter) + '.hdf5')

    start_generator = gen_start_states()
    
    i = 0
    # game_num = 0

    # BASED ON MOVES, WILL BE APPROXIMATE
    moves = 200
    with h5py.File(path, 'w') as f:
        dset = f.create_dataset('neural_games', shape=(moves, 3+(total_squares*3)), dtype=np.int8)

        while i < moves:
            start_state = next(start_generator)
            winner, game, move_len = play_game(size, True, start_state, 0)
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
                i += 1
main()
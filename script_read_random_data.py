import os
import h5py
import random



def collab(p1, p2):
    new = [0 for i in range(9)]
    index = 0

    while index < len(p1):
        if p1[index] == 1:
            new[index] = 1
        index += 1


    index = 0

    while index < len(p2):
        if p2[index] == 1:
            new[index] = 2
        index += 1

    return new


def main():
    size = 3
    path = os.path.join('data', 'random', '3', 'mytestfile0.hdf5')

    total = 0
    with h5py.File(path, 'r') as f:
        moves = f['random_games'].shape[0]

        print('reading {0} moves'.format(moves))

        p1 = 0

        game = 0
        move_counter = 0
        for state in f['random_games']:
            if move_counter == 0:
                if state[0] == 0:
                    print('reached end')
                    break
                game_len = state[2]
                if state[1] == 1:
                    p1 += 1

            # Board stuff
            # print(state[0])
            # print(state[1])
            # print(state[2])

            # move_color = state[0]
            # move_winner= state[1]

            # board_p1 = state[3:3+9]
            # board_p2 = state[12:12+9]

            # move_chosen = state[21:21+9]

            # print(board_p1)
            # print(board_p2)
            # print(move_chosen)

            # full_b = collab(board_p1, board_p2)
            # for i in range(3):
            #     print(full_b[i*3:i*3+3])
            # print(game_len, move_counter)

            # print()


            move_counter += 1
            if move_counter == game_len:
                move_counter = 0
                game += 1
                # exit()
            total += 1

        print('itered {0} moves'.format(total))
        print('read {0} moves'.format(moves))
        print("total games: {0}, total p1 wins: {1}, total p2 wins: {2}, total p1 % winrate {3}%".format(
            game, p1, game-p1, round(p1*100/game, 2)))


    # READING BASED ON GAMES
    # with h5py.File(path, 'r') as f:
    #     games = f['random_games'].shape[0]

    #     print('reading {0} games'.format(games))

    #     p1 = 0
    #     total = 0
    #     for game in f['random_games']:
    #         if game[0][1] == 1:
    #             p1 += 1
    #         total += 1

    #         # Board stuff
    #         for j in range(game[0][2]):
    #             print(game[j][0])
    #             print(game[j][1])
    #             print(game[j][2])

    #             move_color = game[j][0]
    #             move_winner= game[j][1]

    #             board_p1 = game[j][3:3+9]
    #             board_p2 = game[j][12:12+9]

    #             move_chosen = game[j][21:21+9]

    #             print(board_p1)
    #             print(board_p2)
    #             print(move_chosen)

    #             full_b = collab(board_p1, board_p2)
    #             for i in range(3):
    #                 print(full_b[i*3:i*3+3])
    #             print()
    #         break


    #     print("total games: {0}, total p1 wins: {1}, total p2 wins: {2}, total p1 % winrate {3}%".format(
    #         total, p1, total-p1, round(p1*100/total, 2)))

main()

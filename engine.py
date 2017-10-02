import random
import numpy as np

class Engine:
    def __init__(self, n, print_mode, start_state = None):
        self.n = n
        self.board = np.zeros(shape=(9), dtype=int)
        self.track_curr = 0
        self.track = np.zeros(shape=(9), dtype=int)
        self.print_mode = print_mode
        if start_state != None:
            self.init_board(start_state)


    def init_board(self, start_state):
        for i in range(len(start_state)):
            self.board[i] = start_state[i]


    def print_board(self):
        if self.print_mode == 0:
            return
        for x in range(self.n):
            print(self.board[x*self.n:x*self.n+self.n])


    def move(self, x, y, val):
        self.board[y * n + x] = val


    def move(self, m, val):
        self.board[m] = val


    def get_board(self):
        return self.board


    def get_flat_board(self, sym):
        p1 = np.copy(self.board)
        p2 = np.zeros(shape=(self.n*self.n))
        for i in range(len(p1)):
            if p1[1] == sym:
                p1[i] = 1
            else:
                p1[i] = 0
                p2[i] = 1
        return [np.concatenate((p1, p2))]


    def is_terminal(self, moves):
        # left right
        for y in range(self.n):
            for x in range(self.n-2):
                if 0 != self.board[self.to_one_dim(x, y)] == self.board[self.to_one_dim(x+1, y)] == self.board[self.to_one_dim(x+2, y)]:
                    return self.board[self.to_one_dim(x, y)]

        # up down
        for y in range(self.n-2):
            for x in range(self.n):
                if 0 != self.board[self.to_one_dim(x, y)] == self.board[self.to_one_dim(x, y+1)] == self.board[self.to_one_dim(x, y+2)]:
                    return self.board[self.to_one_dim(x, y)]

        # diag right
        for y in range(self.n-2):
            for x in range(self.n-2):
                if 0 != self.board[self.to_one_dim(x, y)] == self.board[self.to_one_dim(x+1, y+1)] == self.board[self.to_one_dim(x+2, y+2)]:
                    return self.board[self.to_one_dim(x, y)]

        # diag left
        for y in range(self.n-2):
            for x in range(self.n-2):
                x = self.n - x - 1
                if 0 != self.board[self.to_one_dim(x, y)] == self.board[self.to_one_dim(x-1, y+1)] == self.board[self.to_one_dim(x-2, y+2)]:
                    return self.board[self.to_one_dim(x, y)]

        if len(moves) == 0:
            return 0
        return -2


    def gen_legal_moves(self):
        for i in range(len(self.board)):
            if self.board[i] == 0:
                yield i


    def get_legal_moves(self):
        a = []
        for i in self.gen_legal_moves():
            a.append(i)
        return a


    def to_one_dim(self, x, y):
        return y * self.n + x


    def push_move(self, x, y, val):
        m = to_one_dim(x, y)
        self.track[self.track_curr] =  m
        self.track_curr += 1
        self.board.move(m, val)


    def pop_move(self):
        self.track_curr -= 1
        self.move(self.track[self.track_curr], 0)
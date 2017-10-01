import random
import tensorflow as tf
import os

class Player:
    def __init__(self, sym, size, engine, print_mode):
        self.sym = sym
        self.size = size
        self.engine = engine
        self.print_mode = print_mode


    def print_turn(self):
        if self.print_mode == 0:
            return
        print("----")
        print("IT IS {0}'s turn".format(self.sym))
        print("----")


    def get_move(self):
        return


    def get_sym(self):
        return self.sym


class Human(Player):
    def __init__(self, n, s, e, print_mode):
        super().__init__(n, s, e, print_mode)


    def get_move(self, moves):
        while True:
            s = "enter number from 0 - {0}:\n".format(self.size-1)
            a = input(s)
            try:
                b = int(a)
            except:
                continue
            if b < 0 or b > self.size-1 or b not in moves:
                continue
            return b


class Random(Player):
    def __init__(self, n, s, e, print_mode):
        super().__init__(n, s, e, print_mode)


    def get_move(self, moves):
        return moves[random.randint(0, len(moves)-1)]


class Network(Player):
    def __init__(self, n, s, e, print_mode, model='policy_net_v2', iteration = float('inf')):
        super().__init__(n, s, e, print_mode)

        model_path = os.path.join('models', model)
        if iteration == float('inf'):
            models = os.listdir(model_path)
            last_iter = len(models) - 1
            model_path = os.path.join(model_path, str(last_iter))
        else:
            model_path = os.path.join(model_path, iteration)

        save_meta_path = os.path.join(model_path, 'ITSYABOI.meta')
        save_ckpt_path = os.path.join(model_path, 'ITSYABOI')
        # saver.restore(sess, save_path)

        self.saver = tf.train.import_meta_graph(save_meta_path)
        self.graph = tf.get_default_graph()
        self.input = self.graph.get_tensor_by_name('Placeholder:0')
        self.compute = self.graph.get_tensor_by_name('add:0')
        self.sess = tf.Session()
        self.saver.restore(self.sess, save_ckpt_path)
        print("model restored: {0}".format(save_ckpt_path))


    def get_move(self, moves):
        model_out = self.compute.eval(session=self.sess, feed_dict={self.input: self.engine.get_flat_board(self.get_sym())})
        model_out = model_out[0]
        pairing = []
        for i in range(len(model_out)):
            pairing.append([model_out[i], i])
        pairing.sort(key=lambda x: x[0])

        for i in pairing:
            if i[1] in moves:
                return i[1]
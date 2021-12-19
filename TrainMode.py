import os
import sys
import time
from collections import deque
from random import shuffle
from Game import Game
from utils.dotdict import dotdict
from pickle import Pickler, Unpickler
from Mcts import Mcts
from NNet import NNet
from PrintBoard import PrintBoard

ff = open("train-output.txt", "a")
BLACK = -2
WHITE = 2
EMPTY = 0
#Everything up-to-date


ARROW = 1


# Parameters of the training mode
args = dotdict({
    'num_iter': 10,            # Number of neural network training
    'num_play_game': 10,       # Play the "num_play_game" game to train NNet once
    'max_len_queue': 200000,   # Maximum length of two-way list
    'num_mcts_search': 1000,   # Number of searches from a state simulation to a leaf node
    'max_batch_size': 20,      # NNet maximum amount of data per training
    'Cpuct': 1,                # The "temperature" hyperparameter in the confidence limit function
    'arenaCompare': 40,
    'tempThreshold': 35,       # Explore Efficiency
    'updateThreshold': 0.55,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/models/', 'best.pth.tar'),
})


class TrainMode:

    def __init__(self, game, nnet):
        """
        :param game: Board Object
        :param nnet: Neural network objects
        """
        self.num_white_win = 0
        self.num_black_win = 0
        self.args = args
        self.player = WHITE
        self.game = game
        self.nnet = nnet
        self.mcts = Mcts(self.game, self.nnet, self.args)
        self.batch = []                 # The amount of data fed to NNet at a time, but not the right type (multidimensional list)
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    # Calling NNet to start training
    def learn(self):
        for i in range(1, self.args.num_iter + 1):
            
            print('')
            
            print('####################################  IterNum: ' + str(i) + ' ####################################')
            
            print('play', self.args.num_play_game, 'games for one NNet training')
            
            print('MCTS number of search simulations:', self.args.num_mcts_search)
            # Execute every time
            if not self.skipFirstSelfPlay or i > 1:
                # deque：Two-way queue  max_len：Maximum Queue Length：self.args.max_len_queue
                # [board, WHITE, pi] data for every trimes training
                iter_train_data = deque([], maxlen=self.args.max_len_queue)

                # play “num_play_game” games for one NNet training
                for j in range(self.args.num_play_game):
                    # Resetting the search tree
                    
                    print("====================================== the", j+1, "th game ======================================")
                    self.mcts = Mcts(self.game, self.nnet, self.args)
                    self.player = WHITE
                    iter_train_data += self.play_one_game()
                    pboard.save_figure(j + 1)
                
                print('TrainMode.py-learn()', 'white wins:', self.num_white_win, 'games；', 'black wins:', self.num_black_win, 'games')
                print('whit score:', self.num_white_win, 'black score:', self.num_black_win, file = ff, flush = True)
                # Print the data given to NN after one iteration
                
                print('TrainMode.py-learn()', len(iter_train_data))
                print('data-size', len(iter_train_data), file = ff, flush = True)
                # save the iteration examples to the history
                self.batch.append(iter_train_data)

            # If the training data is greater than the specified training length, the oldest data is removed
            if len(self.batch) > self.args.max_batch_size:
                print("len(max_batch_size) =", len(self.batch),
                      " => remove the oldest batch")
                self.batch.pop(0)
            
            # Save training data

            self.saveTrainExamples(i - 1)

            # The original batch is a multidimensional list, here the standardized batch
            standard_batch = []
            for e in self.batch:
                # extend() Append multiple elements from other sequences at once at the end of the list
                standard_batch.extend(e)
            # Disrupt the data so that they obey independent identical distribution (exclude correlation between data)
            shuffle(standard_batch)

            # Here a temp is saved, that is, the last network is always saved, here is to play with the latest network
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            # Turn on training
            self.nnet.train(standard_batch)

            print('PITTING AGAINST PREVIOUS VERSION')
            # Number of old and new network wins and ties
            pwins, nwins, draws = 10, 100, 1
            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            # If the sum of the old network and the new network is 0 or if the new network/new network + old network is less than the update threshold (0.55), then no update, otherwise update to the new network parameter
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                # If the new model is rejected, this old model will work
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                # Save the current model and update the latest model
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    # Complete a game
    def play_one_game(self):
        """
        Play a complete game of amazon with Mcts
        :return: 4 * [(board, pi, z)] : Return four training data tuples:（board，strategy，win or lose）
        """
        # [board, WHITE, pi] data of each game
        one_game_train_data = []
        board = self.game.get_init_board(self.game.board_size)
        play_step = 0
        while True:
            play_step += 1
            ts = time.time()
            
            print('---------------------------')
            
            print('the', play_step, 'th step')
            
            print(board)
            pboard.print_board(board, play_step+1)
            self.mcts.episodeStep = play_step
            # In Mcts, always choose from the white perspective
            transformed_board = self.game.get_transformed_board(board, self.player)
            # Probabilities from multiple mcts searches (from white's point of view)
            self.mcts = Mcts(self.game, self.nnet, self.args)
            next_action, steps_train_data = self.mcts.get_best_action(transformed_board)
            one_game_train_data += steps_train_data
            te = time.time()
            if self.player == WHITE:
                
                print("                             white move:", next_action, 'search:', int(te-ts), 's')
            else:
                
                print("                             black move:", next_action, 'search:', int(te - ts), 's')
            board, self.player = self.game.get_next_state(board, self.player, next_action)

            r = self.game.get_game_ended(board, self.player)
            if r != 0:  # r is always -1 when the winner is already decided
                if self.player == WHITE:
                    
                    print('white lose')
                    self.num_black_win += 1
                else:
                    
                    print('black lose')
                    self.num_white_win += 1
                
                print("##### end #####")
                
                print(board)

                pboard.print_board(board, play_step)

                return [(board, pi, r*((-1)**(player != self.player))) for board, player, pi in one_game_train_data]

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.batch)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.batch = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True


if __name__ == "__main__":
    
    board_size_input = int(input("Input game board size: "))
    game = Game(board_size_input)
    nnet = NNet(game)
    train = TrainMode(game, nnet)
    pboard = PrintBoard(game)
    train.learn()

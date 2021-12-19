import numpy as np
import math
from PrintAction import  PrintAction

EPS = 1e-8
BLACK = -2
WHITE = 2
EMPTY = 0
ARROW = 1


class Mcts:
    """
    Monte Carlo tree search class: always use the "white perspective" search to get the next best move for a given board state
    """
    def __init__(self, game, nnet, args):
        """
        :param game: Current board object
        :param nnet: Neural Networks
        :param args: Training parameters
        """
        self.game = game
        self.nnet = nnet
        self.args = args
        self.episodeStep = 0

        self.Game_End = {}        # Win/Loss Status Dictionary
        self.Actions = {}         # All walkable actions in a state
        self.Pi = {}              # Probability of choosing a point during action value: One-dimensional list of 3 * board_size **2 
        self.N = {}               # Number of visits to a state
        self.Nsa = {}             # Number of visits to a state s + action a (the next state) == N[s+1]
        self.Qsa = {}             # Reward value of a state s + action a (the next state)

        self.N_start = {}
        self.N_end = {}
        self.N_arrow = {}
        self.p = PrintAction(self.game)

    def get_best_action(self, board):
        """
        Use white's perspective to determine the best choice
        :param board:  current board
        :return best_action: Next optimal action
        """
        s = self.game.to_string(board)
        for i in range(self.args.num_mcts_search):
            # print('==============================the ', i, ' th search=================================')
            self.search(board)

        # Assertion: The current board is in dictionary N
        assert s in self.N   # Here N is occasionally 1 less than the true number of times 
        # print(self.N[s])
        # print('Mcts-get_best_action: ', self.N[s])
        # If an action exists in the Ns_start record, set the point to Ns_start[(s, a)], else 0
        counts_start = [self.N_start[(s, a)] if (s, a) in self.N_start else 0 for a in range(self.game.board_size**2)]
        # Normalization
        p_start = [x / float(self.N[s]) for x in counts_start]
        counts_end = [self.N_end[(s, a)] if (s, a) in self.N_end else 0 for a in range(self.game.board_size**2)]
        p_end = [x / float(self.N[s]) for x in counts_end]
        counts_arrow = [self.N_arrow[(s, a)] if (s, a) in self.N_arrow else 0 for a in range(self.game.board_size**2)]
        p_arrow = [x / float(self.N[s]) for x in counts_arrow]

        # '''
        #     print
        # '''
        # for i in range(3*self.game.board_size**2):
        #     if i < self.game.board_size**2:
        #         if i == 0:
        #             print('Selecting the Queen's position--------Probability of having been treated by NN----------Exploration times')
        #         print(i, ':------------', self.Pi[s][i], ':-----', counts_start[i])
        #     elif i < 2*self.game.board_size**2:
        #         if i == self.game.board_size**2:
        #             print('Place the Queen--------Probability of having been treated by NN----------Exploration times')
        #         print(i-self.game.board_size**2, ':------------', self.Pi[s][i], ':-----', counts_end[i-self.game.board_size**2])
        #     else:
        #         if i == 2*self.game.board_size**2:
        #             print('Placement of arrows--------Probability of having been treated by NN----------Exploration times')
        #         print(i - 2*self.game.board_size**2, ':------------', self.Pi[s][i], ':-----', counts_arrow[i-2*self.game.board_size**2])

        # Use softmax strategy to select actions
        pi = p_start
        pi = np.append(pi, p_end)
        pi = np.append(pi, p_arrow)

        # Store 4 * [board, WHITE, pi] data for each move
        steps_train_data = []
        # Rotate the position and strategy 180 degrees clockwise to return a tuple of 4 boards and strategies
        sym = self.game.get_symmetries(board, pi)
        for boards, pis in sym:
            steps_train_data.append([boards, WHITE, pis])

        # Select the next step using a probabilistic random strategy
        best_action = self.get_action_on_random_pi(board, pi)
        # Use the value corresponding to the maximum probability for training
        # best_action = self.get_action_on_max_pi(board, pi)
        self.p.print_action(board, pi)
        return best_action, steps_train_data

    def search(self, board):
        """
        Perform a recursive simulation search of the states, adding information about the access nodes of each state (board) (always stored in white's view)
        :param board: current board
        :return: None
        """
        board_copy = np.copy(board)
        board_key = self.game.to_string(board_copy)
        # Determine if the winner has been divided (leaf node)
        if board_key not in self.Game_End:
            self.Game_End[board_key] = self.game.get_game_ended(board_copy, WHITE)

        if self.Game_End[board_key] != 0:
            # print("Simulation to the root node", self.Game_End[board_key])
            return -self.Game_End[board_key]

        # Determine if board_key is a newly expanded node
        if board_key not in self.Pi:
            # by neural net prediction strategy with v([-1,1]) PS[s] as [1:300] array
            self.Pi[board_key], v = self.nnet.predict(board_copy)
            # print(len(self.Pi[board_key]), self.Pi[board_key])
            # print(v)
            # Always look for moves that white can take
            self.Pi[board_key], legal_actions = self.game.get_valid_actions(board_copy, WHITE, self.Pi[board_key])
            # print(legal_actions.shape)
            # Store all possible actions in this state
            self.Actions[board_key] = legal_actions
            self.N[board_key] = 0
            # print('the first time', self.Qsa)
            # Store all possible actions in this state
            self.Actions[board_key] = legal_actions
            self.N[board_key] = 0
            return -v
        legal_actions = self.Actions[board_key]
        best_uct = -float('inf')
        # The best action
        best_action = -1
        psa = list()                  # State transfer probability, length is the number of actions that can be taken in the current state

        # Converts the selection probability Pi into the probability of an action psa
        for a in legal_actions:
            p = 0
            for i in [0, 1, 2]:
                assert self.Pi[board_key][a[i] + i * self.game.board_size ** 2] > 0
                p += math.log(self.Pi[board_key][a[i] + i * self.game.board_size ** 2])
            psa.append(p)
        # print('first print', psa)
        psa = np.array(psa)
        # print('second print', psa)
        psa = np.exp(psa) / sum(np.exp(psa))
        # print('third print', psa)
        # print('------------------------------------------------------------')
        # print(sum(psa), 'Number of selectable actions：', len(psa))   # 近似等于 1
        # Find the upper confidence limit function：Q + Cpuct * p * (Ns square root)/ Nsa
        for i, a in enumerate(legal_actions):              # enumerate():Adding a tuple to a serial number，in which i is serial number：0，1.... a is legal_actions tuple
            if (board_key, a[0], a[1], a[2]) in self.Qsa:  # board_key:board string，a[0], a[1], a[2] is start point, drop point and arrow point 
                u = self.args.Cpuct * psa[i] * math.sqrt(self.N[board_key]) / (1 + self.Nsa[(board_key, a[0], a[1], a[2])])
                # sigmoid_u = 2 * (1/(1+np.exp(-u)) - 0.5)
                uct = self.Qsa[(board_key, a[0], a[1], a[2])] + u
                # print('Traversed actions', a, 'Q value', self.Qsa[(board_key, a[0], a[1], a[2])], 'U value', u, 'UCT', uct)

            else:
                uct = self.args.Cpuct * psa[i] * math.sqrt(self.N[board_key] + EPS)   # Prevent the product from being 0
                # print('Qsa is the u value at 0 point：', uct, a)
            if uct > best_uct:
                best_uct = uct
                best_action = a

        # print('max_uct：', best_uct, 'best_action: ', best_action)
        a = best_action
        # next_player Reversal
        next_board, next_player = self.game.get_next_state(board_copy, WHITE, a)
        # In the next state, reverse the piece colors (next_player = BLACK)
        next_board = self.game.get_transformed_board(next_board, next_player)

        v = self.search(next_board)

        if (board_key, a[0], a[1], a[2]) in self.Qsa:
            self.Qsa[(board_key, a[0], a[1], a[2])] = (self.Nsa[(board_key, a[0], a[1], a[2])] *
                                                       self.Qsa[(board_key, a[0], a[1], a[2])] + v)\
                                                      / (self.Nsa[(board_key, a[0], a[1], a[2])]+1)
            self.Nsa[(board_key, a[0], a[1], a[2])] += 1

        else:
            self.Qsa[(board_key, a[0], a[1], a[2])] = v
            self.Nsa[(board_key, a[0], a[1], a[2])] = 1

        if (board_key, a[0]) in self.N_start:
            self.N_start[(board_key, a[0])] += 1
        else:
            self.N_start[(board_key, a[0])] = 1

        if (board_key, a[1]) in self.N_end:
            self.N_end[(board_key, a[1])] += 1
        else:
            self.N_end[(board_key, a[1])] = 1

        if (board_key, a[2]) in self.N_arrow:
            self.N_arrow[(board_key, a[2])] += 1
        else:
            self.N_arrow[(board_key, a[2])] = 1

        self.N[board_key] += 1

        return -v

    def get_action_on_random_pi(self, board, pi):
        """
        Select the next step using a probabilistic random strategy
        :param board: game board
        :param pi: Overall Probability
        :return best_action: Randomly selected actions based on probability
        """

        pi_start = pi[0:self.game.board_size**2]
        pi_end = pi[self.game.board_size**2:2 * self.game.board_size**2]
        pi_arrow = pi[2 * self.game.board_size**2:3 * self.game.board_size**2]
        # Deep Copy
        copy_board = np.copy(board)
        while True:
            # Pass an array of 1*100 strategy probabilities to get 0~99 action points , action_start,end,arrow are the selected points eg: 43,65....
            action_start = np.random.choice(len(pi_start), p=pi_start)
            # print('start:', action_start)
            action_end = np.random.choice(len(pi_end), p=pi_end)
            # print('end', action_end)
            action_arrow = np.random.choice(len(pi_arrow), p=pi_arrow)
            # print('arrow', action_arrow)
            # Add assertions to ensure that there are discs at the starting point and no discs at the landing and release points
            assert copy_board[action_start // self.game.board_size][action_start % self.game.board_size] == WHITE
            assert copy_board[action_end // self.game.board_size][action_end % self.game.board_size] == EMPTY
            # Cannot assert that the position of the arrow must be empty, it is possible that the position is the queen
            if self.game.is_legal_move(copy_board, action_start, action_end):
                copy_board[action_start // self.game.board_size][action_start % self.game.board_size] = EMPTY
                copy_board[action_end // self.game.board_size][action_end % self.game.board_size] = WHITE
                if self.game.is_legal_move(copy_board, action_end, action_arrow):
                    best_action = [action_start, action_end, action_arrow]
                    # Jumping out of the While loop
                    break
                else:
                    copy_board[action_start // self.game.board_size][action_start % self.game.board_size] = WHITE
                    copy_board[action_end // self.game.board_size][action_end % self.game.board_size] = EMPTY
        return best_action

    def get_action_on_max_pi(self, board, pi):

        poo, legal_actions = self.game.get_valid_actions(board, WHITE, pi)
        # print(pi)
        max_pi = -float('inf')
        best_action = []
        pro = []
        for a in legal_actions:
            p = 0
            for i in [0, 1, 2]:
                # The assertion cannot be added here because Mcts cannot explore all the actions, so it will result in some action points with probability 0
                if pi[a[i] + i * self.game.board_size ** 2] == 0:
                    p = -float('inf')
                    break
                p += math.log(pi[a[i] + i * self.game.board_size ** 2])

            # print(a, np.exp(p) * 100)
            if p > max_pi:
                max_pi = p
                best_action = a
        return best_action

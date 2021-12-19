# coding=utf-8
import numpy as np

BLACK = -2
WHITE = 2
EMPTY = 0
ARROW = 1


class Game:
    directions = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]

    def __init__(self, board_size):
        """
        :param board_size: int: Board size
        :return None
        """
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.get_init_board(board_size)

    def get_init_board(self, board_size):
        """
        :return b.board: Returns a two-dimensional numpy array of boards with board_size*board_size
        """
        self.board = np.zeros((board_size, board_size), dtype=int)
        # Black
        self.board[0][board_size // 3] = BLACK
        self.board[0][2 * board_size // 3] = BLACK
        self.board[board_size // 3][0] = BLACK
        self.board[board_size // 3][board_size - 1] = BLACK
        # white
        self.board[2 * board_size // 3][0] = WHITE
        self.board[2 * board_size // 3][board_size - 1] = WHITE
        self.board[board_size - 1][board_size // 3] = WHITE
        self.board[board_size - 1][2 * board_size // 3] = WHITE
        return self.board

    def get_board_size(self):
        """
        :return (self.board_size, self.board_size): A tuple of board sizes
        """
        return self.board_size, self.board_size

    def get_action_size(self):
        """
        :return 3 * self.board_size * self.board_size:int: Returns the number of spaces for all moves
        """
        return 3 * self.board_size ** 2

    def is_legal_move(self, board, start, end):
        """
        Determine if start->end is moveable
        :param board: Current board
        :param start: Starting point
        :param end: Drop point
        :return: boolean: Movable return True else False
        """
        sx = start // self.board_size  # starting point x
        sy = start % self.board_size  # starting point y
        ex = end // self.board_size  # drop point x
        ey = end % self.board_size  # drop point y
        # print(sx,sy,ex,ey)
        # First determine if the move is along the left/right, up/down, or miko direction, if it is along then it is possible, if it is not then it is never possible and return False
        if ex == sx or ey == sy or abs(ex - sx) == abs(ey - sy):
            tx = (ex - sx) // max(1, abs(ex - sx))  # +1：to right  -1：to left
            ty = (ey - sy) // max(1, abs(ey - sy))  # +1：to up  -1：to down
            t_start = -1  #
            t_end = end  #
            # After that, from the start step to the end, determine whether there are obstacles until the end
            while sx != ex or sy != ey:
                sx += tx
                sy += ty
                t_start = sx * self.board_size + sy
                if board[sx][sy] != EMPTY:  # Not to the end and encounter obstacles
                    break
            if t_start == t_end:  # If the break is due to the smooth walk to the end, then you can go
                if board[sx][sy] != EMPTY:
                    return False
                return True
            else:  # If there is an obstacle and break, then you can not go
                return False
        else:
            return False

    # Update pick,drop,arrow board --> return:new board and next move -----:This method does not discriminate if the start,drop and arrow are in accordance with the rules.
    def get_next_state(self, board, player, action):
        """
        Get the next state
        :param board: n*n board
        :param player:int: current player
        :param action: ternary eg(67,78,99): Start coordinates,queen coordinates,arrow coordinates
        :return board, player
                board: current board
                player: current player: BLACK or WHITE
        """
        b = board.copy()
        start_x, start_y = action[0] // self.board_size, action[0] % self.board_size
        end_x, end_y = action[1] // self.board_size, action[1] % self.board_size
        arrow_x, arrow_y = action[2] // self.board_size, action[2] % self.board_size

        if b[start_x][start_y] != player:
            print("Game-get_next_state: Error Start!", start_x, start_y)
        else:
            b[start_x][start_y] = EMPTY
        if b[end_x][end_y] != EMPTY:
            print("Game-get_next_state: Error End", start_x, start_y)
        else:
            b[end_x][end_y] = player
        if b[arrow_x][arrow_y] != EMPTY:
            print("Game-get_next_state: Error arrow", start_x, start_y)
        else:
            b[arrow_x][arrow_y] = ARROW

        if player == WHITE:
            player = BLACK
        else:
            player = WHITE

        return b, player

    def get_valid_actions(self, board, player, ps):
        """
        Calculate all available moves on the current board, and rearrange the predictions returned by NN: the probability of reaching the most likely point is set to zero
        :param board: n*n board
        :param player:int: current player
        :param ps:3 * board_size ** 2 List: probability values predicted directly using the neural network
        :return: Ps, all_valid_action
                 Ps: Collapsed 3 * board_size ** 2 List;
                 all_valid_action: List composed by ternary (s,e,a): All available moves on the current board
                 Ps -->1 * 300:[0.2， 0.02， 0.23，......]   move:all_valid_action -->[(s, e, a),......]
        """
        b = board.copy()
        # ps = np.copy(ps)
        size = self.board_size  # board size

        ps_start = ps[0:size ** 2]
        ps_end = ps[size ** 2:2 * size ** 2]
        ps_arrow = ps[2 * size ** 2:3 * size ** 2]
        # Define a list of three board lengths：reachable is 1 else 0
        valid_start = np.zeros(size ** 2, dtype=int)
        valid_end = np.zeros(size ** 2, dtype=int)
        valid_arrow = np.zeros(size ** 2, dtype=int)

        all_valid_action = []  # Store the legal moves
        for s in range(size ** 2):  # Pick out the legal moves
            if b[s // size][s % size] == player:
                if self.judge_start_eghit_direction(b,s):
                    valid_start[s] = 1
                    for e in range(size ** 2):
                        if self.is_legal_move(b, s, e):
                            valid_end[e] = 1
                            b[s // size][s % size] = EMPTY
                            b[e // size][e % size] = player
                            for a in range(size ** 2):
                                if self.is_legal_move(b, e, a):
                                    valid_arrow[a] = 1
                                    valid = (s, e, a)
                                    all_valid_action.append(valid)
                            b[s // size][s % size] = player
                            b[e // size][e % size] = EMPTY
        for s in range(size ** 2):
            if valid_start[s] == 0:
                ps_start[s] = 0
        sum_s = np.sum(ps_start)
        if sum_s > 0:
            ps_start /= sum_s
        else:
            print("All start moves were masked, do workaround.")
            # ps_start[valid_start == 1] = 0.25
            for s in range(size ** 2):
                if valid_start[s] == 1:
                    ps_start[s] = 0.25

        for e in range(size ** 2):
            if valid_end[e] == 0:
                ps_end[e] = 0
        sum_e = np.sum(ps_end)
        if sum_e > 0:
            ps_end /= sum_e
        else:
            print("All end moves were masked, do workaround.")

        for a in range(size ** 2):
            if valid_arrow[a] == 0:
                ps_arrow[a] = 0
        sum_a = np.sum(ps_arrow)
        if sum_a > 0:
            ps_arrow /= sum_a
        else:
            print("All arrow moves were masked, do workaround.")

        ps[0:size ** 2] = ps_start
        ps[size ** 2:2 * size ** 2] = ps_end
        ps[2 * size ** 2:3 * size ** 2] = ps_arrow

        all_valid_action = np.array(all_valid_action)  # Convert to an array
        return ps, all_valid_action

    def get_game_ended(self, board, player):
        """
        Determine if the game ends when the player's turn comes to make a move
        :param board: n*n board
        :param player: Which player's turn it is to make a move
        :return: 0 or -1: 0—>Fail to determine win or lose；-1—>player lose :Failure to determine a win (judging in advance that your opponent can't move is sometimes inaccurate; your opponent may be able to move again after your move)
        """
        size = self.board_size
        # Record the number of player's moveable pieces
        count_player = 0
        # Record the number of pieces available to the player's opponent
        # count_opposite_player = 0
        for i in range(size):
            for j in range(size):
                if board[i][j] == player:
                    # Determine if a move is possible for eight directions
                    for k in range(8):
                        m = i + self.directions[k][0]
                        n = j + self.directions[k][1]
                        if m not in range(size) or n not in range(size):
                            continue
                        if board[m][n] == EMPTY:
                            count_player += 1
                            break
                # if board[i][j] == -player:
                #     # Determine if a move is possible for eight directions
                #     for k in range(8):
                #         m = i + self.directions[k][0]
                #         n = j + self.directions[k][1]
                #         if m not in range(size) or n not in range(size):
                #             continue
                #         if board[m][n] == EMPTY:
                #             count_opposite_player += 1
                #             break
        # If the current player's moveable pieces are 0, then the current player loses
        if count_player == 0:
            return -1
        # # If the current opponent's moveable pieces are 0, then the current player wins
        # if count_opposite_player == 0:
        #     return 1
        return 0

    def get_symmetries(self, board, pi):
        """
        Rotate the position and strategy 180 degrees clockwise to return a tuple of 4 boards and strategies, with the aim of increasing the amount of data used to train the neural network
        :param board: n*n board
        :param pi: 3 * board_size ** 2 List:Strategy vector
        :return: board_list: four (board, strategy) tuple
        """
        pi_board_start = np.reshape(pi[0:self.board_size**2], (self.board_size, self.board_size))
        pi_board_end = np.reshape(pi[self.board_size**2:2 * self.board_size**2], (self.board_size, self.board_size))
        pi_board_arrow = np.reshape(pi[2 * self.board_size**2:3 * self.board_size**2], (self.board_size, self.board_size))
        board_list = []
        board_list += [(board, pi)]  # Each board and an array of 1*300 strategies, forming a tuple

        # Flip both the board and the strategy vector two-dimensional arrays left and right
        newB = np.fliplr(board)
        newPi_start = np.fliplr(pi_board_start)
        newPi_end = np.fliplr(pi_board_end)
        newPi_arrow = np.fliplr(pi_board_arrow)
        newPi = newPi_start
        newPi = np.append(newPi, newPi_end)
        newPi = np.append(newPi, newPi_arrow)
        board_list += [(newB, list(newPi.ravel()))]  # ravel()：The function of converting a multi-dimensional array to a one-dimensional array

        # Reverse the order of the boards and the strategy vectors in the first dimension.
        newB = np.flipud(board)
        newPi_start = np.flipud(pi_board_start)
        newPi_end = np.flipud(pi_board_end)
        newPi_arrow = np.flipud(pi_board_arrow)
        newPi = newPi_start
        newPi = np.append(newPi, newPi_end)
        newPi = np.append(newPi, newPi_arrow)
        board_list += [(newB, list(newPi.ravel()))]

        # Flip the two-dimensional arrays of board and strategy vectors left and right again
        newB = np.fliplr(newB)
        newPi_start = np.fliplr(newPi_start)
        newPi_end = np.fliplr(newPi_end)
        newPi_arrow = np.fliplr(newPi_arrow)
        newPi = newPi_start
        newPi = np.append(newPi, newPi_end)
        newPi = np.append(newPi, newPi_arrow)
        board_list += [(newB, list(newPi.ravel()))]
        return board_list

    def get_transformed_board(self, board, player):
        """
        Always turn the intelligence to the white view when playing from the game, used in MCTS search to store each state property
        :param board: n*n board
        :param player: current player
        :return board: Converted board
        """
        # Deep Copy
        if player == WHITE:
            return board
        board = np.copy(board)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == WHITE:
                    board[i][j] = BLACK
                elif board[i][j] == BLACK:
                    board[i][j] = WHITE
        return board

    def judge_start_eghit_direction(self, board, s):
        # Prevent array overruns by first determining
        for a in self.directions:
            if (s // self.board_size + a[0]) >= 0 and (s % self.board_size + a[1]) >= 0 and (s // self.board_size + a[0]) < self.board_size and (s % self.board_size + a[1]) < self.board_size:
                if board[s // self.board_size + a[0]][s % self.board_size + a[1]] == 0:
                    return True
        return False

    @staticmethod
    def to_string(board):
        """
        Convert the board to a string, in preparation for using the board as a key for the dictionary later
        :param board: n*n board
        :return: str string
        """
        return board.tostring()


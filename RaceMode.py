import statistics
import time
import copy
from collections import deque

import pygame
import sys

f = open("myprint.txt", "a")
global DEEP_MAX, node_act, action_player1, action_player2, tip_estimate
game_start_time = int(round(time.time() * 1000))

def check_distance(r1, c1, list_int, condition):
    """
    The function checks whether the distance between a square and at least one item in a list of squares is less than or equal to a value.
     Arguments:
         r1 (int): the index of the row on which the square is located
         c1 (int): the index of the column on which the square is located
         list_int [(int, int)]: list of pairs of integers
         condition (int): the value with which the distance is compared
    """
    if condition == -1:
        return True
    for (r2, c2) in list_int:
        if abs(r1 - r2) + abs(c1 - c2) <= condition:
            return True
    return False

class Game:
    '''
    The class that defines the game
    '''
    gmin = None
    gmax = None
    GOL = '▢'
    ARROW = 'X'
    row = None
    column = None
    max_score = 0

    def __init__(self, matrix=None, row=None, column=None):
        if matrix:
            # the game already started
            self.matrix = matrix
        else:
            # initialize the game
            self.matrix = [[self.__class__.GOL] * column for i in range(row)]
            self.matrix[0][3] = "B"
            self.matrix[0][6] = "B"
            self.matrix[3][0] = "B"
            self.matrix[3][9] = "B"
            self.matrix[6][0] = "W"
            self.matrix[9][3] = "W"
            self.matrix[9][6] = "W"
            self.matrix[6][9] = "W"

            if row is not None:
                self.__class__.row = row
            if column is not None:
                self.__class__.column = column
            
            #calculate maximum score
            score_row = (column - 10) * row
            score_column = (row - 10) * column
            score_diagonal = (row - 10) * (column - 10) * 2
            self.__class__.max_score = score_row + score_column + score_diagonal
    
    def draw_grid(self, selected=None):
        for index in range(self.__class__.column * self.__class__.row):
            row_g = index // self.__class__.column
            column_g = index % self.__class__.column

            if index == selected:
                color = (0,100,100) # the color for the selected square
            else:
                # the white or black color on the squares
                if(column_g + row_g) % 2 == 0:
                    color = (255,255,255)
                else:
                    color = (0,0,0)
            pygame.draw.rect(self.__class__.display, color, self.__class__.grid[index])
            if self.matrix[row_g][column_g] == 'B':
                self.__class__.display.blit(self.__class__.black_image, (
                    column_g * (self.__class__.dark_square + 1), row_g * (self.__class__.dark_square + 1)))
            elif self.matrix[row_g][column_g] == 'W':
                self.__class__.display.blit(self.__class__.white_image, (
                    column_g * (self.__class__.dark_square + 1), row_g * (self.__class__.dark_square + 1)))
            elif self.matrix[row_g][column_g] == Game.ARROW:
                self.__class__.display.blit(self.__class__.x_image, (
                    column_g * (self.__class__.dark_square + 1), row_g * (self.__class__.dark_square + 1)))
        pygame.display.flip()
    
    @classmethod
    def counter_player(c_game, player):
        return c_game.gmax if player == c_game.gmin else c_game.gmin
    
    @classmethod
    def initialize(c_game, display, row=10, column=10, dark_square=50):
        c_game.display = display
        c_game.dark_square = dark_square
        c_game.black_image = pygame.image.load('blackPiece.png')
        c_game.black_image = pygame.transform.scale(c_game.black_image, (dark_square, dark_square))
        c_game.white_image = pygame.image.load('whitePiece.png')
        c_game.white_image = pygame.transform.scale(c_game.white_image, (dark_square, dark_square))
        c_game.x_image = pygame.image.load('arrowImage.png')
        c_game.x_image = pygame.transform.scale(c_game.x_image, (dark_square, dark_square))

        c_game.grid = [] # the list of squares in the grid

        for row_g in range(row):
            for column_g in range(column):
                pattern = pygame.Rect(column_g * (dark_square + 1), row_g * (dark_square + 1), dark_square, dark_square)
                c_game.grid.append(pattern)
    
    def display_information(self, player):
        '''
        Function for displaying the player whose turn it is and displaying the possibility to stop the game
        Arguments:
             player (str): the player whose turn it is
        '''
        player = 'W' if player == 'W' else 'B'
        text1 = "Current Turn"
        font = pygame.font.Font('freesansbold.ttf',25)
        text1 = font.render(text1, True, (0,0,250), (250,100,250))
        textRect = text1.get_rect()
        textRect.center = (655,250)
        self.__class__.display.blit(text1, textRect)
        if player == "B":
            text2 = "Black"
        else:
            text2 = "White"
        font = pygame.font.Font('freesansbold.ttf', 25)
        text2 = font.render(text2, True, (0, 0, 250), (250, 100, 250))
        textRect = text2.get_rect()
        textRect.center = (650, 275)
        self.__class__.display.blit(text2, textRect)
        text4 = "Press Q to Exit"
        font = pygame.font.Font('freesansbold.ttf', 16)
        text4 = font.render(text4, True, (0, 0, 250), (250, 100, 250))
        textRect = text4.get_rect()
        textRect.center = (662, 305)
        self.__class__.display.blit(text4, textRect)
        text5 = "Press S to Switch Turn"
        font = pygame.font.Font('freesansbold.ttf', 16)
        text5 = font.render(text5, True, (0, 0, 250), (250, 100, 250))
        textRect = text5.get_rect()
        textRect.center = (662, 335)
        self.__class__.display.blit(text5, textRect)
        pygame.display.flip()
    
    def check_able_to_move_piece(self, player):
        '''
        Function that checks if, for a player, at least one piece can be moved
        Arguments:
             player (str): the player for whom the check is performed
        '''
        piecePositionList = self.get_piece_position(player)
        move = [(1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
        for (r,c) in piecePositionList:
            for (mr,mc) in move:
                if 0 <= r + mr <= self.row -1 and 0 <= c + mc <= self.column -1:
                    if self.matrix[r + mr][c + mc] == Game.GOL:
                        return True # at least one piece can be moved
        return False
    
    def final(self):
        '''
        Function that checks if a game has reached the end (when a player can no longer move any pieces)
        '''
        if not self.check_able_to_move_piece(Game.gmin) and not self.check_able_to_move_piece(Game.gmax):
            # no player can move
            return "TIE"
        if not self.check_able_to_move_piece(Game.gmin):
            # gmin can't move
            return Game.gmax
        if not self.check_able_to_move_piece(Game.gmax):
            # gmax can't move
            return Game.gmin
        return False

    def occupied_square_num(self):
        '''
        Function that calculates the number of occupied cells in the game board
        '''
        num = 0
        for i in range(self.row):
            for j in range(self.column):
                if self.matrix[i][j] != Game.GOL:
                    num += 1
        return num

    def get_piece_position(self, player):
        '''
        For a player, the list of cell positions on which the player's pieces are located is returned
        Arguments:
             player (str): the player for whom the position of the pieces is searched
        '''
        positionList = []
        numPiece = 0
        for i in range(self.__class__.row):
            for j in range(self.__class__.column):
                if numPiece == 4:
                    return positionList
                if self.matrix[i][j] == player:
                    positionList.append((i, j))
                    numPiece += 1
        return positionList
    
    def place_X(self, r_n, c_n, opponent_position):
        '''
        Function that performs the placements of ARROW after moving a piece.
         If only 20 squares in the game board are occupied, then the ARROW will be placed at a distance of 1 from at least one
         opponent's piece. If only 30 squares in the game board are occupied then the ARROW will be placed at a distance of 2
         from at least one piece of the opponent.
         All directions to the part in which the ARROW can be placed (row, column, diagonal) are traversed and the positions
         of the squares that can be reached and are unoccupied are retained.
         Arguments:
             r_n (int): the row on which the moved piece for which the ARROW placement is desired is located
             c_n (int): the column on which the moved piece is located for which the Arrow placement is desired
             opponent_position [(int, int)]: the list of positions of the opponent's pieces
        '''
        occupied_square = self.occupied_square_num()
        if occupied_square <= 20:
            distance = 1
        elif occupied_square <=30:
            distance = 2
        else:
            distance = -1
        possible_position = []
        #finding the positions in which the X can be placed
        for j in range(c_n + 1, self.__class__.column):
            if self.matrix[r_n][j] == Game.GOL and check_distance(r_n, j, opponent_position, distance):
                possible_position.append((r_n, c_n, r_n, j))
            else:
                break
        for j in range(c_n-1, -1, -1):
            if self.matrix[r_n][j] == Game.GOL and check_distance(r_n, j, opponent_position, distance):
                possible_position.append((r_n, c_n, r_n, j))
            else:
                break
        for i in range(r_n + 1, self.__class__.row):
            if self.matrix[i][c_n] == Game.GOL and check_distance(i, c_n, opponent_position, distance):
                possible_position.append((r_n, c_n, i, c_n))
            else:
                break
        for i in range(r_n-1, -1, -1):
            if self.matrix[i][c_n] == Game.GOL and check_distance(i, c_n, opponent_position, distance):
                possible_position.append((r_n, c_n, i, c_n))
            else:
                break
        i = r_n + 1
        j = c_n + 1
        while i < self.__class__.row and j < self. __class__.column:
            if self.matrix[i][j] == Game.GOL and check_distance(i, j, opponent_position, distance):
                possible_position.append((r_n, c_n, i, j))
                i += 1
                j += 1
            else:
                break
        i = r_n - 1
        j = c_n - 1
        while i >= 0 and j >= 0:
            if self.matrix[i][j] == Game.GOL and check_distance(i, j, opponent_position, distance):
                possible_position.append((r_n, c_n, i, j))
                i -= 1
                j -= 1
            else:
                break
        i = r_n - 1
        j = c_n + 1
        while i >= 0 and j < self.__class__.column:
            if self.matrix[i][j] == Game.GOL and check_distance(i, j, opponent_position, distance):
                possible_position.append((r_n, c_n, i, j))
                i -= 1
                j += 1
            else:
                break
        i = r_n + 1
        j = c_n - 1
        while i < self.__class__.row and j >= 0:
            if self.matrix[i][j] == Game.GOL and check_distance(i, j, opponent_position, distance):
                possible_position.append((r_n, c_n, i, j))
                i += 1
                j -= 1
            else:
                break
        return possible_position
    
    def move(self, player):
        '''
        The function of generating moves for a player
        First we look for all the positions in which the player can move each piece as follows:
             for each piece of the player the rows / columns / diagonals of the piece are traversed and for each
             accessible and unoccupied square both the initial position of the piece and the newly found position
             are retained.
        For each new part placement position, the ARROW placement positions are searched
        Arguments:
            player (str): the player for whom the moves are sought
        '''
        piecePositionList = self.get_piece_position(player)
        counter_player = self.counter_player(player)
        piecePositionList_cp = self.get_piece_position(counter_player)
        possible_position = []
        # finding the positions in which the player can move a piece keeping in mind the position of the piece
        # from which he started
        for (r, c) in piecePositionList:
            for j in range(c + 1, self.__class__.column):
                if self.matrix[r][j] == Game.GOL:
                    possible_position.append((r, c, r, j))
                else:
                    break
            for j in range(c - 1, -1, -1):
                if self.matrix[r][j] == Game.GOL:
                    possible_position.append((r, c, r, j))
                else:
                    break
            for i in range(r + 1, self.__class__.row):
                if self.matrix[i][c] == Game.GOL:
                    possible_position.append((r, c, i, c))
                else:
                    break
            for i in range(r - 1, -1 -1):
                if self.matrix[i][c] == Game.GOL:
                    possible_position.append((r, c, i, c))
                else:
                    break
            i = r + 1
            j = c + 1
            while i < self.__class__.row and j < self.__class__.column:
                if self.matrix[i][j] == Game.GOL:
                    possible_position.append((r, c, i, j))
                    i += 1
                    j += 1
                else:
                    break
            i = r - 1
            j = c - 1
            while i >= 0 and j >= 0:
                if self.matrix[i][j] == Game.GOL:
                    possible_position.append((r, c, i, j))
                    i -= 1
                    j -= 1
                else:
                    break
            i = r - 1
            j = c + 1
            while i >= 0 and j < self.__class__.column:
                if self.matrix[i][j] == Game.GOL:
                    possible_position.append((r, c, i, j))
                    i -= 1
                    j += 1
                else:
                    break
            i = r + 1
            j = c - 1
            while i < self.__class__.row and j >= 0:
                if self.matrix[i][j] == Game.GOL:
                    possible_position.append((r, c, i, j))
                    i += 1
                    j -= 1
                else:
                    break
        move_list = []
        for p in possible_position:
            new_matrix = copy.deepcopy(self.matrix)
            (old_r_piece, old_c_piece, new_r_piece, new_c_piece) = p
            new_matrix[old_r_piece][old_c_piece] = Game.GOL
            new_matrix[new_r_piece][new_c_piece] = player
            newGame = Game(new_matrix)
            # for each move of the piece, we also place the X
            possible_position_X = newGame.place_X(new_r_piece, new_c_piece,piecePositionList_cp)
            for pos in possible_position_X:
                (new_r_piece_X, new_c_piece_X, r_X, c_X) = pos
                new_matrix = copy.deepcopy(newGame.matrix)
                new_matrix[r_X][c_X] = Game.ARROW
                newGameX = Game(new_matrix)
                move_list.append(newGameX)
        return move_list
    
    def accessible_square(self, player):
        '''
        Function that counts the accessible squares in the player's pieces
        Arguments:
             player (str): the player for whom the accessible squares are searched
        '''
        num = 0
        piecePositionList = self.get_piece_position(player)
        for i in range(self.__class__.row):
            for j in range(self.__class__.column):
                if self.matrix[i][j] == Game.GOL:
                    for (r, c) in piecePositionList:
                        if self.valid_X_move(r, c, i, j):
                            # if the cell on row i and column j is accessible to a player's piece, the number of
                            # accessible blocks increases
                            num += 1
                            break
        return num

    def perform_move_matrix(self, player):
        '''
        Function that performs the move matrix for a player move_matrix[i][j] = the minimum number of moves from
        the player's pieces to the square on row i and column j
        '''
        move_matrix = [[float('inf') for i in range(self.column)] for j in range(self.row)]
        piecePositionList = self.get_piece_position(player)
        for (r, c) in piecePositionList:
            move_matrix[r][c] = 0
        piecePosition = deque(piecePositionList)
        while piecePosition:
            (r, c) = piecePosition.popleft()
            for j in range(c + 1, self.__class__.column):
                if self.matrix[r][j] == Game.GOL:
                    if move_matrix[r][c] + 1 < move_matrix[r][j]:  # if a shorter path was found
                        move_matrix[r][j] = move_matrix[r][c] + 1
                        piecePosition.append((r, j))
                else:
                    break
            for j in range(c - 1, -1, -1):
                if self.matrix[r][j] == Game.GOL:
                    if move_matrix[r][c] + 1 < move_matrix[r][j]:  # if a shorter path was found
                        move_matrix[r][j] = move_matrix[r][c] + 1
                        piecePosition.append((r, j))
                else:
                    break
            for i in range(r + 1, self.__class__.row):
                if self.matrix[i][c] == Game.GOL:
                    if move_matrix[r][c] + 1 < move_matrix[r][c]:  # if a shorter path was found
                        move_matrix[i][c] = move_matrix[r][c] + 1
                        piecePosition.append((i, c))
                else:
                    break
            for i in range(r - 1, -1, -1):
                if self.matrix[i][c] == Game.GOL:
                    if move_matrix[r][c] + 1 < move_matrix[i][c]:  # if a shorter path was found
                        move_matrix[i][c] = move_matrix[r][c] + 1
                        piecePosition.append((i, c))
                else:
                    break
            i = r + 1
            j = c + 1
            while i < self.__class__.row and j < self.__class__.column:  # if a shorter path was found
                if self.matrix[i][j] == Game.GOL:
                    if move_matrix[r][c] + 1 < move_matrix[i][j]:
                        move_matrix[i][j] = move_matrix[r][c] + 1
                        piecePosition.append((i, j))
                    i += 1
                    j += 1
                else:
                    break
            i = r - 1
            j = c - 1
            while i >= 0 and j >= 0:
                if self.matrix[i][j] == Game.GOL:
                    if move_matrix[r][c] + 1 < move_matrix[i][j]:  # if a shorter path was found
                        move_matrix[i][j] = move_matrix[r][c] + 1
                        piecePosition.append((i, j))
                    i -= 1
                    j -= 1
                else:
                    break
            i = r - 1
            j = c + 1
            while i >= 0 and j < self.__class__.column:
                if self.matrix[i][j] == Game.GOL:
                    if move_matrix[r][c] + 1 < move_matrix[i][j]:  # if a shorter path was found
                        move_matrix[i][j] = move_matrix[r][c] + 1
                        piecePosition.append((i, j))
                    i -= 1
                    j += 1
                else:
                    break
            i = r + 1
            j = c - 1
            while i < self.__class__.row and j >= 0:
                if self.matrix[i][j] == Game.GOL:
                    if move_matrix[r][c] + 1 < move_matrix[i][j]:  # if a shorter path was found
                        move_matrix[i][j] = move_matrix[r][c] + 1
                        piecePosition.append((i, j))
                    i += 1
                    j -= 1
                else:
                    break
        return move_matrix

    def square_num_difference(self, player):
        """
        Function that returns the difference between the squares that "belong" to the player and the squares that "belong"
        to the opposite player
        A square "belongs" to a player if he has the number of moves to the square smaller than the opponent's
        """
        num = 0
        num_cp = 0
        counter_player = self.counter_player(player)
        move_matrix = self.perform_move_matrix(player)
        move_matrix_cp = self.perform_move_matrix(counter_player)
        # returns the number of cells the player reaches faster than his opponent
        for i in range(self.row):
            for j in range(self.column):
                if move_matrix[i][j] == move_matrix_cp[i][j]:
                    continue  # cell is neutral
                if move_matrix[i][j] < move_matrix_cp[i][j]:
                    num += 1  # the cell is closer to the player
                else:
                    num_cp += 1
        return num-num_cp

    def score_estimate(self, depth):
        """
        Because the goal of the game is to block the opponent and avoid own blocking, two ways of estimating the
        score were used:
             - the number of blocks accessible for MAX - the number of blocks accessible for MIN (the more MAX has more
             blocks accessible than MIN, the more it means that MIN is more blocked than MAX,
             which is an advantage for MAX)
             - squares that "belong" to MAX - squares that "belong" to MIN: a square "belongs" to a player if he can reach it
             in a smaller number of moves than the opponent (when a player has faster access in one square than the other,
             that square is a place where the player cannot be blocked immediately by the opponent;
             the more MAX square there are, the harder it is to block)
        """
        global tip_estimate
        t_final = self.final()
        if t_final == self.__class__.gmax:
            return self.__class__.max_score + depth
        elif t_final == self.__class__.gmin:
            return -self.__class__.max_score - depth
        elif t_final == 'Tie':
            return 0
        else:
            if tip_estimate == "1":
                return self.accessible_square(self.__class__.gmax) - self.accessible_square(self.__class__.gmin)
            else:
                return self.accessible_square(self.__class__.gmax)
    
    def valid_X_move(self, r_old, c_old, r_new, c_new):
        """
        Function that checks if an ARROW part / placement move is valid (valid_X_move)
        A move of piece / placement of ARROW is valid if from the initial piece it is possible to reach the new position on
        the row, column or diagonal, without encountering occupied cells on the road
         Arguments:
             r_old (int): the index of the square row on which the original piece is located
             c_old (int): the index of the square column on which the original piece is located
             r_new (int): index of the square row on which the moved piece is located / ARROW placed
             c_new (int): index of the square row on which the moved piece is located / ARROW placed
        """
        move_X = True
        if r_old == r_new and c_old == c_new:
            return False
        for j in range(c_old + 1, self.__class__.column):
            if self.matrix[r_old][j] != Game.GOL:
                move_X = False
            if r_old == r_new and j == c_new and self.matrix[r_old][j] == Game.GOL:
                return move_X  # if there was an element between the original part / X and the new
                # position then it is not a correct X move or placement, otherwise it is correct
        move_X = True
        for j in range(c_old - 1, -1, -1):
            if self.matrix[r_old][j] != Game.GOL:
                move_X = False
            if r_old == r_new and j == c_new and self.matrix[r_old][j] == Game.GOL:
                return move_X  # if there was an element between the original part / X and the new
                # position then it is not a correct X move or placement, otherwise it is correct
        move_X = True
        for i in range(r_old + 1, self.__class__.row):
            if self.matrix[i][c_old] != Game.GOL:
                move_X = False
            if r_new == i and c_old == c_new and self.matrix[i][c_old] == Game.GOL:
                return move_X  # if there was an element between the original part / X and the new
                # position then it is not a correct X move or placement, otherwise it is correct
        move_X = True
        for i in range(r_old - 1, -1, -1):
            if self.matrix[i][c_old] != Game.GOL:
                move_X = False
            if r_new == i and c_old == c_new and self.matrix[i][c_old] == Game.GOL:
                return move_X  # if there was an element between the original part / X and the new
                # position then it is not a correct X move or placement, otherwise it is correct
        i = r_old + 1
        j = c_old + 1
        move_X = True
        while i < self.__class__.row and j < self.__class__.column:
            if self.matrix[i][j] != Game.GOL:
                move_X = False
            if i == r_new and j == c_new and self.matrix[i][j] == Game.GOL:
                return move_X  # if there was an element between the original part / X and the new
                # position then it is not a correct X move or placement, otherwise it is correct
            i += 1
            j += 1
        i = r_old - 1
        j = c_old - 1
        move_X = True
        while i >= 0 and j >= 0:
            if self.matrix[i][j] != Game.GOL:
                move_X = False
            if i == r_new and j == c_new and self.matrix[i][j] == Game.GOL:
                return move_X  # if there was an element between the original part / X and the new
                # position then it is not a correct X move or placement, otherwise it is correct
            i -= 1
            j -= 1
        i = r_old - 1
        j = c_old + 1
        move_X = True
        while i >= 0 and j < self.__class__.column:
            if self.matrix[i][j] != Game.GOL:
                move_X = False
            if i == r_new and j == c_new and self.matrix[i][j] == Game.GOL:
                return move_X  # if there was an element between the original part / X and the new
                # position then it is not a correct X move or placement, otherwise it is correct
            i -= 1
            j += 1
        i = r_old + 1
        j = c_old - 1
        move_X = True
        while i < self.__class__.row and j >= 0:
            if self.matrix[i][j] != Game.GOL:
                move_X = False
            if i == r_new and j == c_new and self.matrix[i][j] == Game.GOL:
                return move_X  # if there was an element between the original part / X and the new
                # position then it is not a correct X move or placement, otherwise it is correct
            i += 1
            j -= 1
        return False  # it is not a correct piece move or X placement
    
    def color_final(self, winner):
        """
        Function that colors at the end of the game the squares on which the winner's pieces are
        """
        if winner != "Tie":
            pieceList = self.get_piece_position(winner)
            for index in range(self.__class__.column * self.__class__.row):
                row_g = index // self.__class__.column
                column_g = index % self.__class__.column
                for (r, c) in pieceList:
                    if r == row_g and c == column_g:
                        pygame.draw.rect(self.__class__.display, (100, 250, 100), self.__class__.grid[index])
                        if self.matrix[row_g][column_g] == winner and winner == 'B':
                            self.__class__.display.blit(self.__class__.black_image,
                                                        (column_g * (self.__class__.dark_square + 1),
                                                         row_g * (self.__class__.dark_square + 1)))
                        elif self.matrix[row_g][column_g] == winner and winner == 'W':
                            self.__class__.display.blit(self.__class__.white_image, (
                                column_g * (self.__class__.dark_square + 1), row_g * (self.__class__.dark_square + 1)))
            pygame.display.flip()
        
    def final_display(self, winner, time_vector, node_vector):
        """
        Function that displays information about a game when it is over
        Arguments:
             winner (str): the player who won
             time_vector (int list): the vector that contains the computer's thinking times
             node_vector (int list): the vector that contains the number of nodes generated by MinMax / AlphaBeta at
                                       every move
        """
        global action_player1, action_player2, game_start_time
        if time_vector and node_vector:
            print("Minimum AI thinking time: ", min(time_vector))
            print("Maximum AI thinking time: ", max(time_vector))
            print("Average AI thinking time: ", round(sum(time_vector) / len(time_vector)))
            print("Median AI thinking time: ", statistics.median(time_vector))

            print("Total number of nodes generate:", sum(node_vector), file = f, flush = True)
            print("Total number of nodes generate:", sum(node_vector), file = sys.stdout)
            print("Minimum number of nodes generated for each move: ", min(node_vector))
            print("Maximum number of nodes generated for each move: ", max(node_vector))
            print("Average number of nodes generated for each move: ", round(sum(node_vector) / len(node_vector)))
            print("Median number of nodes generated for each move: ", statistics.median(node_vector))
        print("Total number of AI moves: ", action_player2)
        print("The total number of moves the player makes: ", action_player1)
        game_end_time = int(round(time.time() * 1000))
        print("The game lasted: ", game_end_time - game_start_time, " milliseconds")
        if winner == -1:
            return
        self.color_final(winner)
        winner = "White" if winner == "W" else "BLACK"
        time.sleep(3)
        self.__class__.display.fill((96, 213, 121))
        if winner == "Tie":
            text = "It was a tie!"
        else:
            text = "Winner is " + winner
        font = pygame.font.Font('freesansbold.ttf', 50)
        text = font.render(text, True, (0, 0, 250), (250, 100, 250))
        textRect = text.get_rect()
        textRect.center = (375, 250)
        self.__class__.display.blit(text, textRect)
        button = ButtonGroup(
            top=300,
            left=350,
            buttonList=[
                Button(display=self.__class__.display, w=100, h=40, text="Quit game", value="Quit"),
            ], selectfigure=0)
        button.depict()
        pygame.display.flip()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if button.mouseSelect(pos):
                        pygame.quit()
                        sys.exit()
                    else:
                        break
                else:
                    break

    def strgDiplay(self):
        strg = "  |"
        strg += " ".join([str(i) for i in range(self.column)]) + "\n"
        strg += "-" * (self.column + 1) * 2 + "\n"
        strg += "\n".join([str(i) + " |" + " ".join([str(x) for x in self.matrix[i]]) for i in range(len(self.matrix))])
        return strg

    def __str__(self):
        return self.strgDiplay()

    def __repr__(self):
        return self.strgDiplay()

class AI:
    """
    The class used by the minimax and alpha-beta algorithms
    It owns the game board
    """

    def __init__(self, game_table, game_current, depth, module=None, score=None):
        self.game_table = game_table
        self.game_current = game_current

        # depth in the state tree
        self.depth = depth

        # the score of the state (if it is the final) or of the best state-child (for the current player)
        self.score = score

        # list of possible moves from the current state
        self.possible_move = []

        # the best move from the list of possible moves for the current player
        self.best_move = None

    def move(self):
        moveList = self.game_table.move(self.game_current)
        counter_player = Game.counter_player(self.game_current)
        AI_move = [AI(movements, counter_player, self.depth - 1, module=self) for movements in moveList]
        return AI_move

    def __str__(self):
        strg = str(self.game_table) + "(Current Game:" + self.game_current + ")\n"
        return strg

    def __repr__(self):
        strg = str(self.game_table) + "(Current Game:" + self.game_current + ")\n"
        return strg

# Algoritmul MinMax
def min_max(ai):
    global node_act
    node_act += 1

    if ai.depth == 0 or ai.game_table.final():
        ai.score = ai.game_table.score_estimate(ai.depth)
        return ai

    # calculate all possible moves from the current state
    ai.possible_move = ai.move()

    # applying the minimax algorithm to all possible moves
    move_score = [min_max(movement) for movement in ai.possible_move]

    if ai.game_current == Game.gmax:
        # if the player is gmax, the child status with the maximum score is chosen
        ai.best_move = max(move_score, key=lambda x: x.score)
    else:
        # if the player is gmin, the child status with the minimum score is chosen
        ai.best_move = min(move_score, key=lambda x: x.score)
    ai.score = ai.best_move.score
    return ai

# Algoritmul AlphaBeta
def alpha_beta(alpha, beta, ai):
    global node_act
    node_act += 1

    if ai.depth == 0 or ai.game_table.final():
        ai.score = ai.game_table.score_estimate(ai.depth)
        return ai

    if alpha > beta:
        return ai  # it is in an invalid range so I don't process it anymore

    ai.possible_move = ai.move()

    if ai.game_current == Game.gmax:
        current_score = float('-inf')

        for movements in ai.possible_move:
            # calculate the score
            new_ai = alpha_beta(alpha, beta, movements)

            if current_score < new_ai.score:
                ai.best_move = new_ai
                current_score = new_ai.score
            if alpha < new_ai.score:
                alpha = new_ai.score
                if alpha >= beta:
                    break

    elif ai.game_current == Game.gmin:
        current_score = float('inf')

        for movements in ai.possible_move:

            new_ai = alpha_beta(alpha, beta, movements)

            if current_score > new_ai.score:
                ai.best_move = new_ai
                current_score = new_ai.score

            if beta > new_ai.score:
                beta = new_ai.score
                if alpha >= beta:
                    break
    ai.score = ai.best_move.score

    return ai

def show_final(current_ai):
    """
    Function that checks if the current state is final state   
    """
    final = current_ai.game_table.final()
    if final:
        if final == "Tie":
            print("Tie")
        else:
            print("Won " + final)
        return final
    return False

class Button:#Button
    def __init__(self, display=None, left=0, top=0, w=0, h=0, backgroundColor=(93, 153, 162),
                 backgroundColorSelect=(250, 100, 0),
                 text="", font="arial", fontSize=16, textColor=(255, 255, 255), value=""):
        self.display = display
        self.backgroundColor = backgroundColor
        self.backgroundColorSelect = backgroundColorSelect
        self.text = text
        self.font = font
        self.w = w
        self.h = h
        self.select = False
        self.fontSize = fontSize
        self.textColor = textColor
        # the font object is created
        fontObject = pygame.font.SysFont(self.font, self.fontSize)
        self.textFrame = fontObject.render(self.text, True, self.textColor)
        self.rectangle = pygame.Rect(left, top, w, h)
        # text center
        self.textRectangle = self.textFrame.get_rect(center=self.rectangle.center)
        self.value = value

    def selection(self, selected):
        self.select = selected
        self.depict()#depict

    def mouseSelect(self, helper):#mouseSelect
        if self.rectangle.collidepoint(helper):
            self.selection(True)
            return True
        return False

    def updateRectangle(self):
        self.rectangle.left = self.left
        self.rectangle.top = self.top
        self.textRectangle = self.textFrame.get_rect(center=self.rectangle.center)

    def depict(self):#depict
        bColor = self.backgroundColorSelect if self.select else self.backgroundColor
        pygame.draw.rect(self.display, bColor, self.rectangle)
        self.display.blit(self.textFrame, self.textRectangle)

class ButtonGroup:#ButtonGroup
    def __init__(self, buttonList=[], selectfigure=0, buttonSpace=10, left=0, top=0):
        self.buttonList = buttonList
        self.selectfigure = selectfigure
        self.buttonList[self.selectfigure].select = True
        self.top = top
        self.left = left
        leftCurrent = self.left
        for b in self.buttonList:
            b.top = self.top
            b.left = leftCurrent
            b.updateRectangle()
            leftCurrent += (buttonSpace + b.w)

    def mouseSelect(self, helper):
        for ib, b in enumerate(self.buttonList):
            if b.mouseSelect(helper):
                self.buttonList[self.selectfigure].selection(False)
                self.selectfigure = ib
                return True
        return False

    def depict(self):
        for b in self.buttonList:
            b.depict()

    def getValue(self):
        return self.buttonList[self.selectfigure].value

def function_draw(display, game_table):
    b_algorithm = ButtonGroup(
        top=30,
        left=30,
        buttonList=[
            Button(display=display, w=100, h=40, text="Minimax", value="minimax"),
            Button(display=display, w=100, h=40, text="Alphabeta", value="alphabeta")
        ],
        selectfigure=0)
    b_piece = ButtonGroup(
        top=100,
        left=30,
        buttonList=[
            Button(display=display, w=100, h=40, text="Black pieces", value="B"),
            Button(display=display, w=100, h=40, text="White pieces", value="W")
        ],
        selectfigure=0)
    b_difficulty = ButtonGroup(
        top=170,
        left=30,
        buttonList=[
            Button(display=display, w=120, h=40, text="Beginner difficulty", value="db"),
            Button(display=display, w=120, h=40, text="Medium difficulty", value="dm"),
            Button(display=display, w=120, h=40, text="Advanced difficulty", value="da"),
        ],
        selectfigure=0)
    b_estimate = ButtonGroup(
        top=240,
        left=30,
        buttonList=[
            Button(display=display, w=120, h=40, text="Estimation 1", value="1"),
            Button(display=display, w=120, h=40, text="Estimation 2", value="2"),
        ],
        selectfigure=0)
    start = Button(display=display, top=310, left=80, w=80, h=40, text="Start game", backgroundColor=(250, 0, 0))
    b_algorithm.depict()
    b_piece.depict()
    b_difficulty.depict()
    b_estimate.depict()
    start.depict()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if not b_algorithm.mouseSelect(pos):
                    if not b_piece.mouseSelect(pos):
                        if not b_difficulty.mouseSelect(pos):
                            if not b_estimate.mouseSelect(pos):
                                if start.mouseSelect(pos):
                                    display.fill((0, 0, 0))
                                    game_table.draw_grid()
                                    return b_piece.getValue(), b_algorithm.getValue(), b_difficulty.getValue(), b_estimate.getValue()
        pygame.display.update()

def main():
    global DEEP_MAX, action_player1, action_player2, tip_estimate

    pygame.init()
    pygame.display.set_caption("Game of the Amazons")
    # window size in pixels
    r = 10
    c = 10
    w = 50
    ecran = pygame.display.set_mode(size=(c * (w + 1) - 1 + 300, r * (w + 1) - 1))
    Game.initialize(ecran, row=r, column=c, dark_square=w)

    # board initialization
    current_table = Game(row=10, column=10)
    Game.gmin, tip_algorithm, difficulty, tip_estimate = function_draw(ecran, current_table)
    print("Selected parameters: ", Game.gmin, tip_algorithm, difficulty, tip_estimate)

    if difficulty == "db":
        DEEP_MAX = 1
    elif difficulty == "dm":
        DEEP_MAX = 2
    else:
        DEEP_MAX = 3

    # the computer plays with the remaining color
    Game.gmax = 'W' if Game.gmin == 'B' else 'B'

    print("Initial table")
    print(str(current_table))

    # initial state creation - white moves first
    current_state = AI(current_table, 'W', DEEP_MAX)

    act_move = False
    act_select_X = False
    piece_move = False
    piece_position = False
    current_table.draw_grid()
    time_vector = []
    node_vector = []
    action_player1 = 0
    action_player2 = 0
    t_initial = int(round(time.time() * 1000))
    while True:
        break_flag = False
        if current_state.game_current == Game.gmin:

            Game.display_information(current_state.game_table, Game.gmin)
            if break_flag:
                break
            # it's the player's turn
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYDOWN:
                    pressed = pygame.key.get_pressed()
                    if pressed[pygame.K_q]:
                        Game.final_display(current_state.game_table, -1, time_vector, node_vector)
                        pygame.quit()
                        sys.exit()
                    elif pressed[pygame.K_s]:
                        current_state.game_current = Game.counter_player(current_state.game_current)
                        break_flag = True
                        break

                elif event.type == pygame.MOUSEBUTTONDOWN:

                    pos = pygame.mouse.get_pos()

                    for np in range(len(Game.grid)):

                        if Game.grid[np].collidepoint(
                                pos):  # check if the coordinate point is in the rectangle (square)
                            row_g = np // 10
                            column_g = np % 10

                            if current_state.game_table.matrix[row_g][column_g] == Game.gmin and not piece_move:
                                if act_move and row_g == act_move[0] and column_g == act_move[1]:
                                    # if the piece was selected it is selected
                                    act_move = False
                                    current_state.game_table.draw_grid()
                                else:
                                    # a piece is selected
                                    act_move = (row_g, column_g)
                                    current_state.game_table.draw_grid(np)

                            if current_state.game_table.matrix[row_g][column_g] == Game.GOL and act_move:
                                # if a piece is selected then it can be moved
                                if Game.valid_X_move(current_state.game_table, act_move[0], act_move[1], row_g,
                                                               column_g):
                                    current_state.game_table.matrix[act_move[0]][act_move[1]] = Game.GOL
                                    current_state.game_table.matrix[row_g][column_g] = Game.gmin
                                    act_move = False
                                    act_select_X = True
                                    piece_move = True
                                    piece_position = (row_g, column_g)
                                    current_state.game_table.draw_grid()

                            if current_state.game_table.matrix[row_g][column_g] == Game.GOL and act_select_X:
                                # if a piece has been moved, the X can be placed
                                if Game.valid_X_move(current_state.game_table, piece_position[0],
                                                               piece_position[1],
                                                               row_g, column_g):
                                    current_state.game_table.matrix[row_g][column_g] = Game.ARROW
                                    act_select_X = False
                                    piece_move = False

                                    # displaying the status of the game after player move
                                    print("\nThe board after moving the player")
                                    print(str(current_state))

                                    current_state.game_table.draw_grid()

                                    t_end = int(round(time.time() * 1000))
                                    print("The player thought for " + str(t_end - t_initial) + " milliseconds.")
                                    action_player1 += 1
                                    # checking if a final state has been reached
                                    result = show_final(current_state)
                                    if result:
                                        Game.final_display(current_state.game_table, result, time_vector,
                                                                     node_vector)
                                        break

                                    # a move was made; the player changes
                                    current_state.game_current = Game.counter_player(current_state.game_current)

        else:
            Game.display_information(current_state.game_table, Game.gmax)

            global node_act
            node_act = 0

            t_initial = int(round(time.time() * 1000))
            if tip_algorithm == 'minimax':
                update_state = min_max(current_state)
            else:
                update_state = alpha_beta(-500, 500, current_state)
            current_state.game_table = update_state.best_move.game_table
            print("The board after moving the AI")
            print(str(current_state))

            current_state.game_table.draw_grid()
            # preiau timpul in milisecunde de dupa mutare
            t_end = int(round(time.time() * 1000))
            print("The player thought for " + str(t_end - t_initial) + " milliseconds.")
            print("They were generated ", node_act, " nodes when moving")
            print("Estimating the current state is ", update_state.score)

            node_vector.append(node_act)
            time_vector.append(t_end - t_initial)
            action_player2 += 1

            result = show_final(current_state)
            if result:
                Game.final_display(current_state.game_table, result, time_vector, node_vector)
                break

            # a move was made; the player changes
            current_state.game_current = Game.counter_player(current_state.game_current)
            t_initial = int(round(time.time() * 1000))

if __name__ == "__main__":
    main()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
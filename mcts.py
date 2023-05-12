import random
import math

class TicTacToe:
    def __init__(self):
        self.board = [[' ' for _ in range(4)] for _ in range(4)]
        self.current_player = 'X'
        self.status = 'ongoing'
        self.winning_combinations = [
            [(0,0), (0,1), (0,2), (0,3)], [(1,0), (1,1), (1,2), (1,3)], [(2,0), (2,1), (2,2), (2,3)], [(3,0), (3,1), (3,2), (3,3)],
            [(0,0), (1,0), (2,0), (3,0)], [(0,1), (1,1), (2,1), (3,1)], [(0,2), (1,2), (2,2), (3,2)], [(0,3), (1,3), (2,3), (3,3)],
            [(0,0), (1,1), (2,2), (3,3)], [(0,3), (1,2), (2,1), (3,0)]
        ]

    def print_board(self):
        print('  1 2 3 4')
        for i in range(4):
            row = self.board[i]
            print(f'{i+1} {" ".join(row)}')

    def make_move(self, x, y):
        if self.board[x][y] == ' ':
            self.board[x][y] = self.current_player
            self.check_win()
            self.current_player = 'X' if self.current_player == 'O' else 'O'
            return True
        else:
            return False

    def check_win(self):
        for combination in self.winning_combinations:
            if all(self.board[x][y] == self.current_player for x, y in combination):
                self.status = f'{self.current_player} wins'
        if all(self.board[i][j] != ' ' for i in range(4) for j in range(4)) and self.status == 'ongoing':
            self.status = 'draw'

class Node:
    def __init__(self, parent=None, move=None, player=None):
        self.parent = parent
        self.move = move
        self.player = player
        self.wins = 0
        self.simulations = 0
        self.children = []

    def add_child(self, move, player):
        # Creates a new child node with the given state and appends the new node to the children list of the current node.
        child = Node(parent=self, move=move, player=player)
        self.children.append(child)
        return child

    def ucb_score(self, C): #calculates the score for a child node based on number of wins and visits, and C
        #formula is wins / visits + c * sqrt(log(parent_visits) / visits)
        if self.simulations == 0:
            return float('inf')
        exploitation = self.wins / self.simulations
        exploration = math.sqrt(math.log(self.parent.simulations) / self.simulations)
        #return exploitation + self.C * exploration
        return exploitation + C * exploration


class MonteCarloTreeSearch:
    def __init__(self, game, C=10):
        self.game = game
        self.C = C

    def search(self, num_simulations=30000):
        # uses state and position to find best move
        root = Node()
        for _ in range(num_simulations):
            node = root
            while node.children:
                node = max(node.children, key=lambda n: n.ucb_score(self.C))
            if node.simulations == 0:
                self.expand(node)
            else:
                self.simulate(node)
        return max(root.children, key=lambda n: n.simulations)

    def expand(self, node):
        # takes node and creates child nodes for all possible moves from the state represented by the given node
        possible_moves = [(x, y) for x in range(4) for y in range(4) if self.game.board[x][y] == ' ']
        for move in possible_moves:
            node.add_child(move, self.game.current_player)

        child_node = random.choice(node.children)
        result = self.simulate(child_node)
        self.backpropagate(child_node, result)

    def simulate(self, node):
        # simulates the game until it ends from the given node
        game_copy = TicTacToe()
        game_copy.board = [row[:] for row in self.game.board]
        game_copy.current_player = self.game.current_player
        game_copy.status = self.game.status

        current_node = node
        while game_copy.status == 'ongoing':
            if current_node.children:
                current_node = max(current_node.children, key=lambda n: n.ucb_score(self.C))
                x, y = current_node.move
                game_copy.make_move(x, y)
            else:
                possible_moves = [(x, y) for x in range(4) for y in range(4) if game_copy.board[x][y] == ' ']
                if possible_moves:
                    x, y = random.choice(possible_moves)
                    game_copy.make_move(x, y)

        if game_copy.status == f'{node.player} wins':
            return 1
        elif game_copy.status == 'draw':
            return 0.5
        else:
            return 0

    def backpropagate(self, node, result): #result should be win: 1, loss: -1, and tie: 0
        # Updates the wins and visits counters for each node in the tree from the given node up to the root node, based on the outcome of a simulation. 
        while node is not None:
            node.simulations += 1
            node.wins += result if node.player == self.game.current_player else 1 - result
            node = node.parent

def get_human_move():
    while True:
        move = input('Enter your move in the format row,column (e.g. 1,3): ')
        try:
            x, y = move.split(',')
            x, y = int(x) - 1, int(y) - 1
            return x, y
        except ValueError:
            print('Invalid format. Please enter row,column (e.g. 1,3).')

def main():
    game = TicTacToe()
    mcts = MonteCarloTreeSearch(game)
    while game.status == 'ongoing':
        if game.current_player == 'X':
            x, y = mcts.search().move
            print(f'AI plays: {x+1},{y+1}')
            game.make_move(x, y)
        else:
            game.print_board()
            x, y = get_human_move()
            while not game.make_move(x, y):
                print('Invalid move.')
                x, y = get_human_move()
        print()
    game.print_board()
    print(game.status)

if __name__ == '__main__':
    main()

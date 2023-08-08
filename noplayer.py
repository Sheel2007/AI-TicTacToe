import random
import math
import matplotlib.pyplot as plt

class TicTacToe:
    def __init__(self, size=4):
        self.size = size
        self.board = [[' ' for _ in range(size)] for _ in range(size)]
        self.current_player = 'X'
        self.status = 'ongoing'
        self.winning_combinations = self.generate_winning_combinations()

    def generate_winning_combinations(self):
        # Generate winning combinations for rows, columns, and diagonals.
        winning_combinations = []

        # Rows and Columns
        for i in range(self.size):
            winning_combinations.append([(i, j) for j in range(self.size)])
            winning_combinations.append([(j, i) for j in range(self.size)])

        # Diagonals
        winning_combinations.append([(i, i) for i in range(self.size)])
        winning_combinations.append([(i, self.size - 1 - i) for i in range(self.size)])

        return winning_combinations

    def print_board(self):
        print('  ' + ' '.join(str(i) for i in range(1, self.size + 1)))
        for i in range(self.size):
            row = self.board[i]
            print(f'{i+1} {" ".join(row)}')

    def make_move(self, x, y):
        #  - checks if a move is valid
        #    - switches player turn and returns true
        #  - otherwise returns false
        if self.board[x][y] == ' ':
            self.board[x][y] = self.current_player
            self.check_win()
            self.current_player = 'X' if self.current_player == 'O' else 'O'
            return True
        else:
            return False

    def check_win(self):
        #checks for game ending status (pretty self explanatory)
        for combination in self.winning_combinations:
            if all(self.board[x][y] == self.current_player for x, y in combination):
                self.status = f'{self.current_player} wins'
        if all(self.board[i][j] != ' ' for i in range(self.size) for j in range(self.size)) and self.status == 'ongoing':
            self.status = 'draw'

class Node:
    """
     - Represents a node in the Monte Carlo Tree Search (MCTS) tree.
     - Each node has the game board and player at that state, along with stats (# of wins & # of sims) for the MCTS algorithm.
     - Also maintains a list of child nodes which represent possible moves.
    """
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

    def ucb_score(self, C): 
        # Upper Confidence Bound (UCB1) scores for a child node based on number of wins and visits, and C
        # ucb = wins / visits + c * sqrt(log(parent_visits) / visits)
        # or alternatively ucb = exploitation (win rate) + c * exploration
        # c is a tunable weight to determine the level of exploration
        # exploitaion is already know nodes to reach win
        # exploration is exploring new nodes to reach win
        
        if self.simulations == 0:
            return float('inf')
        exploitation = self.wins / self.simulations # basically the win rate
        exploration = math.sqrt(math.log(self.parent.simulations) / self.simulations)
        #return exploitation + self.C * exploration
        return exploitation + C * exploration
    # the idea behind this is to determine which child node to explore further based on their win rate
    # https://towardsdatascience.com/monte-carlo-tree-search-in-reinforcement-learning-b97d3e743d0f 



class MonteCarloTreeSearch:
    def __init__(self, game, C=0.07):
        self.game = game
        self.C = C

    def search(self, num_simulations=2000):
        # Searches for the best move by performing a number of sims.
        # Returns the best move node.

        root = Node() # this node represents the current game status

        #perform mcts by running sims
        for _ in range(num_simulations):
            node = root
            while node.children:
                # Select the child node with the highest UCB1 score
                node = max(node.children, key=lambda n: n.ucb_score(self.C))
            if node.simulations == 0:
                 # If the node hasn't been simulated yet, expand it by adding child nodes for all possible moves.
                self.expand(node)
            else:
                # Otherwise, perform a simulation from the selected node
                self.simulate(node)

        # Return the child node with the most visited (highest # of sims) as the best move.
        return max(root.children, key=lambda n: n.simulations)

    def expand(self, node):
        # takes node and creates child nodes for all possible moves from the state represented by the given node
        possible_moves = [(x, y) for x in range(self.game.size) for y in range(self.game.size) if self.game.board[x][y] == ' ']
        for move in possible_moves:
            node.add_child(move, self.game.current_player)
        
        # Choose a random child node for simulation.
        child_node = random.choice(node.children)
        result = self.simulate(child_node)
        self.backpropagate(child_node, result)

    def simulate(self, node):
        # Simulates the game until it reaches an end state for a node.
        # Returns the result of the sim.
        game_copy = TicTacToe(self.game.size)
        game_copy.board = [row[:] for row in self.game.board]
        game_copy.current_player = self.game.current_player
        game_copy.status = self.game.status

        current_node = node
        while game_copy.status == 'ongoing':
            if current_node.children:
                # Select the child node with the highest UCB1 score for exploration/exploitation.
                current_node = max(current_node.children, key=lambda n: n.ucb_score(self.C))
                x, y = current_node.move
                game_copy.make_move(x, y) #makes the "best" move
            else:
                possible_moves = [(x, y) for x in range(game_copy.size) for y in range(game_copy.size) if game_copy.board[x][y] == ' ']
                if possible_moves:
                    x, y = random.choice(possible_moves)
                    game_copy.make_move(x, y)

        if game_copy.status == f'{node.player} wins':
            return 1
        elif game_copy.status == 'draw':
            return 0.5
        else:
            return 0

    def backpropagate(self, node, result): 
        # Updates the wins and sims for each node in the tree from the given node up to the root node.
        # The result of the sim is 1 for win, 0.5 for draw, and 0 for loss
        while node is not None:
            node.simulations += 1
            node.wins += result if node.player == self.game.current_player else 1 - result
            node = node.parent

def play_games(num_games=100):
    game_nums = []
    win_rates = []

    # Create the figure and axes for the plot
    fig, ax = plt.subplots()
    ax.set_xlabel('Game')
    ax.set_ylabel('Average Win Rate')
    ax.set_title(f'Average Win Rate Over {num_games} Games')

    for game_num in range(1, num_games + 1):
        print(f"Playing game {game_num}...")

        game = TicTacToe()
        mcts = MonteCarloTreeSearch(game)

        win_count_x = 0

        for _ in range(game.size * game.size):
            if game.current_player == 'X':
                x, y = mcts.search().move
            else:
                possible_moves = [(x, y) for x in range(game.size) for y in range(game.size) if game.board[x][y] == ' ']
                x, y = random.choice(possible_moves)

            game.make_move(x, y)

            if game.status == f'X wins':
                win_count_x += 1
                break

        win_rate_x = win_count_x / game_num
        win_rates.append(win_rate_x)
        game_nums.append(game_num)

        if game_num % 10 == 0:
            avg_win_rate = sum(win_rates[-10:]) / 10
            print(f"Average win rate over last 10 games: {avg_win_rate:.2f}")

            # Update the data on the plot
            ax.clear()
            ax.plot(game_nums, win_rates)
            ax.set_xlabel('Game')
            ax.set_ylabel('Average Win Rate')
            ax.set_title(f'Average Win Rate Over {num_games} Games')

            # Pause to allow time for display
            plt.pause(0.01)

    plt.show()

if __name__ == '__main__':
    num_games = 100
    play_games(num_games)
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
# import modin.experimental.pandas as pd # Use optimized pandas wrapper for multi-threading processing?
# import modin.pandas as mpd

import chess
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import IterableDataset
# from torch.utils.data import DataLoader
# import glob


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(8,8))#, stride=1, padding=1) # kernel size of (8,8)
        self.fc1 = nn.Linear(64 + 2, 1024)  # 64 input nodes + 1 input node for turn + 1 node for check
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 64*64)  # 64*64 output nodes

    def forward(self, x, scalar_inputs):
        x = x.view(x.size(0), 3, 8, 8)
        x = nn.LeakyReLU()(self.conv1(x))
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = torch.cat((x, scalar_inputs), dim=1)
        x = nn.Tanh()(self.fc1(x))
        x = nn.LeakyReLU()(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, 64, 64)  # Reshape the output to (batch_size, 64, 64)


def is_legal_move(fen, move):
    if move[:2] == move[2:]:
        return False

    board = chess.Board(fen)
    move = chess.Move.from_uci(move)
    return move in board.legal_moves


def is_check(fen):
    board = chess.Board(fen)
    return board.is_check()


def custom_loss(output, target, fen, illegal_move_penalty=250.0):
    # Calculate the MSE loss
    output = output.view(-1, 64, 64)
    
    # Replae MSE loss with L1Loss
    # loss = nn.MSELoss()(output, target)

    # L1 Norm for Loss
    loss = nn.L1Loss()(output, target)

    # Add penalty for illegal moves
    batch_size = output.size(0)
    penalties = torch.zeros(batch_size)
    for i in range(batch_size):
        move = output_to_move(output[i])
        if not is_legal_move(fen[i], move):
            penalties[i] = illegal_move_penalty

    loss += penalties.mean()

    return loss




def possible_move_ending_positions(fen):
    board = chess.Board(fen)
    all_moves = list(board.legal_moves)
    possible_ending_positions = [m.uci()[2:] for m in all_moves]
    board_tensor = torch.zeros(64)
    for square in possible_ending_positions:
        board_tensor[square_to_index(square)] = 1
    return board_tensor

def possible_move_starting_positions(fen):
    board = chess.Board(fen)
    all_moves = list(board.legal_moves)
    possible_starting_positions = [m.uci()[:2] for m in all_moves]
    board_tensor = torch.zeros(64)
    for square in possible_starting_positions:
        board_tensor[square_to_index(square)] = 1
    return board_tensor


def train(model, criterion, optimizer, num_epochs):
    print('Begin Training!')
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        try:
            csv_files = get_csv_files("../Data/DataTrain")
            current_csv = np.random.choice(csv_files)
            df = pd.read_csv(current_csv, dtype={'move': 'str', 'cp': 'str', 'cp_rel': 'str', 'cp_loss': 'str', 'is_blunder_cp': 'str'}, chunksize=8000)
            count = 0
            # print(data.head())

            for data in df:
                count += 1                
                col_board = data['board']
                col_white_to_move = data['white_active']
                col_move = data['move']
                col_is_check = data['is_check']

                # board_tensor = torch.zeros(len(col_board), 64)
                # move_tensor = torch.zeros(len(col_move), 64, 64)

                board_tensor = torch.stack([fen_to_tensor(fen) for fen in col_board])
                move_tensor = torch.stack([move_to_tensor(move) for move in col_move])

                board_tensor = board_tensor.to(device)
                move_tensor = move_tensor.to(device)
                white_to_move_tensor = torch.tensor(col_white_to_move.values).to(device)
                is_check_tensor = torch.tensor(col_is_check.values).to(device)
                
                possible_move_ending_positions_tensor = torch.stack([possible_move_ending_positions(fen) for fen in col_board]).to(device)
                possible_move_starting_positions_tensor = torch.stack([possible_move_starting_positions(fen) for fen in col_board]).to(device)

                scalar_inputs = torch.cat((white_to_move_tensor.unsqueeze(1), is_check_tensor.unsqueeze(1)), dim=1).to(device)

                inputs = torch.concat((board_tensor, possible_move_ending_positions_tensor, possible_move_starting_positions_tensor), dim=1).to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs, scalar_inputs)
                loss = criterion(outputs, move_tensor, col_board.values)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                print(f'\tEpoch {epoch+1} Batch {count} Loss: {loss.item():.4f}')

        except KeyboardInterrupt:
            print("Finished Training Early!")
            break
        finally:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/count:.4f}')

    print('Finished Training!')

    torch.save(model, 'models/autosave.pth')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def modify_output(output):
    # Add a small random noise to the output tensor
    noise = torch.randn_like(output) * 0.1
    output += noise

    return output


def predict(model, fen):
    # Convert the FEN string to a tensor
    white_to_move = int(chess.Board(fen).turn)

    fen_tensor = fen_to_tensor(fen).to(device)

    move_tensor = torch.tensor(white_to_move).unsqueeze(0).to(device)
    check_tensor = torch.tensor(is_check(fen)).unsqueeze(0).to(device)

    scalar_inputs = torch.cat((move_tensor, check_tensor), dim=0).T

    model.eval()

    possible_move_ending_positions_tensor = torch.stack(possible_move_ending_positions(fen)).to(device)
    possible_move_starting_positions_tensor = torch.stack(possible_move_starting_positions(fen)).to(device)

    inputs = torch.concat((fen_tensor, possible_move_ending_positions_tensor, possible_move_starting_positions_tensor), dim=1).to(device)

    # Get the model's predictions
    with torch.no_grad():
        output = model(inputs, scalar_inputs.unsqueeze(0))

    output = output.cpu() # Move tensor back to cpu

    # Convert the output tensor to a move
    move = output_to_move(output[0])

    # If the move is illegal, modify the output tensor to suggest a different move
    while not is_legal_move(fen, move):
        print(f'UserWarning: Illegal move {move}, modifying output tensor...')
        
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        random_move = np.random.choice(legal_moves)
        move = random_move.uci()

    return move



def move_to_tensor(move):
    # Convert the move to a source and destination square
    source_square = move[:2]
    dest_square = move[2:]

    # Convert the squares to indices
    source_index = square_to_index(source_square)
    dest_index = square_to_index(dest_square)

    move_tensor = torch.zeros(64, 64)
    move_tensor[source_index][dest_index] = 1

    return move_tensor

def square_to_index(square):
    rank = 8 - int(square[1])  # Ranks numbered 8 to 1
    file = ord(square[0]) - ord('a')  # Files lettered a to h
    return rank * 8 + file

def output_to_move(output):
    output_matrix = output.view(64, 64)

    # Find indices of maximum val in the output matrix
    source_index, dest_index = torch.unravel_index(torch.argmax(output_matrix), output_matrix.shape)

    # Convert the indices to a move in algebraic notation
    move = index_to_square(source_index.item()) + index_to_square(dest_index.item())

    return move

def fen_to_tensor(fen):
    # Define a mapping from pieces to integers
    piece_to_int = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,
        '.': 0
    }

    # Split the FEN string into parts ## 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    parts = fen.split(' ')
    ranks = parts[0].split('/') # Only process the board position (the first part)

    # Convert the ranks to a list of integers
    board = []
    for rank in ranks:
        for char in rank:
            if char.isdigit():
                # If the character is a digit, add that many zeros to the board
                board.extend([0] * int(char))
            else:
                # Otherwise, add the integer representation of the piece to the board
                board.append(piece_to_int[char])

    # Convert the board to a tensor
    board_tensor = torch.tensor(board, dtype=torch.float32)

    return board_tensor

def fen_list_to_tensor(fen_list):
    samples = []
    for fen in fen_list:
        # Define a mapping from pieces to integers
        piece_to_int = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,
            '.': 0
        }

        # Split the FEN string into parts ## 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        parts = fen.split(' ')
        ranks = parts[0].split('/') # Only process the board position (the first part)

        # Convert the ranks to a list of integers
        board = []
        for rank in ranks:
            for char in rank:
                if char.isdigit():
                    # If the character is a digit, add that many zeros to the board
                    board.extend([0] * int(char))
                else:
                    # Otherwise, add the integer representation of the piece to the board
                    board.append(piece_to_int[char])

        # Convert the board to a tensor
        board_tensor = torch.tensor(board, dtype=torch.float32)

        samples.append(board_tensor)
    return torch.tensor(samples, dtype=torch.float32)


def modify_output(output):
    # Add a small random noise to the output tensor
    noise = torch.randn_like(output) * 0.01
    output += noise

    return output

def index_to_square(index):
    # Convert the index to a rank and file
    rank = 8 - (index // 8)  # Ranks are numbered from 8 to 1
    file = chr((index % 8) + ord('a'))  # Files are lettered from a to h

    # Return the square in algebraic notation
    return file + str(rank)


def get_csv_files(path):
    import glob
    csv_files = glob.glob(f'{path}/*')
    return csv_files

def main():
    path = "../Data/DataTrain"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from torch.utils.data import DataLoader
    

    # Get a list of all CSV files in the directory
    csv_files = get_csv_files(path)

    # Create a dataset
    dataset = ChessIterableDataset(csv_files, chunksize=50000)

    # Create a data loader
    data_loader = DataLoader(dataset, batch_size=150)

    model = Net()
    model = model.to(device)

    criterion = custom_loss
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    train(model, custom_loss, optimizer, num_epochs=100000)

    torch.save(model, './models/chess_SL_model_V7.pth')


# if __name__ == '__main__':
#     main()
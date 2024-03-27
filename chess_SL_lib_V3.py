import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import modin.experimental.pandas as pd # Use optimized pandas wrapper for multi-threading processing?
import modin.pandas as mpd

import chess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
import glob


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ChessIterableDataset(IterableDataset): # TODO! Write docstrings 

    def __init__(self, csv_files, chunksize):
        self.csv_files = csv_files
        self.chunksize = chunksize

    def process_chunk(self, chunk):
        # Process the chunk and return a list of samples
        samples = []
        for index, row in chunk.iterrows():
            
            # Feature Variables
            board = row['board']
            white_to_move = row['white_active']
            cp_rel = row['cp_rel'] # add to predictions? would need to get from stockfish
            is_check = row['is_check'] # add to predictions

            # Predictor Variable
            move = row['move']

            # Convert data to tensors
            board_tensor = self.fen_to_tensor(board)
            white_to_move = torch.tensor(white_to_move, dtype=torch.float32)
            cp_rel = torch.tensor(cp_rel, dtype=torch.float32)
            is_check = torch.tensor(is_check, dtype=torch.float32)
            move = self.move_to_tensor(move)

            samples.append({'fen': board_tensor, 'fen_str': board, 'white_to_move': white_to_move, 'cp_rel': cp_rel, 'is_check': is_check, 'move': move})
        return samples

    def __len__(self):
        return sum(1 for _ in self.__iter__())


    def __getitem__(self, idx):
        board = self.dataframe.iloc[idx]['board']
        white_to_move = self.dataframe.iloc[idx]['white_active']
        cp_rel = self.dataframe.iloc[idx]['cp_rel']
        is_check = self.dataframe.iloc[idx]['is_check']
        move = self.dataframe.iloc[idx]['move']

        # Convert data to tensors
        board_tensor = self.fen_to_tensor(board)
        white_to_move = torch.tensor(white_to_move, dtype=torch.float32)
        cp_rel = torch.tensor(cp_rel, dtype=torch.float32)
        move = self.move_to_tensor(move)

        return {'fen': board_tensor, 'fen_str': board, 'white_to_move': white_to_move, 'cp_rel': cp_rel, 'is_check': is_check, 'move': move}

    
    def __iter__(self):
        for idx, csv_file in enumerate(self.csv_files):
            chunk_iter = pd.read_csv(csv_file, chunksize=self.chunksize, dtype={15:str, 33:str})#, engine='pyarrow')
            
            if idx % 10 == 0:
                print(f'Read file {idx} of {len(self.csv_files)}')

            for chunk in chunk_iter:
                
                chunk = chunk.loc[chunk['white_elo'] >= 1350] # select a subset of players with a certain rating
                chunk = chunk.loc[chunk['black_elo'] >= 1350]
                yield from self.process_chunk(chunk) # Returns a list of yielded elements
            
            del chunk_iter # clean up garbage
    
    '''
    # modin implementation
    def __iter__(self):
        for idx, csv_file in enumerate(self.csv_files):
            # chunk_iter = pd.read_csv(csv_file, chunksize=self.chunksize, engine='pyarrow')
            chunk = mpd.read_csv(csv_file, dtype={15: str, 33: str})
            
            if idx % 5 == 0:
                print(f'Read file {idx} of {len(self.csv_files)}')

            chunk = chunk.loc[chunk['white_elo'] >= 1350] # select a subset of players with a certain rating
            chunk = chunk.loc[chunk['black_elo'] >= 1350]
            yield from self.process_chunk(chunk) # Returns a list of yielded elements
            
            del chunk # clean up garbage
    '''
    
    def move_to_tensor(self, move):
        # Convert the move to a source and destination square
        source_square = move[:2]
        dest_square = move[2:]

        # Convert the squares to indices
        source_index = self.square_to_index(source_square)
        dest_index = self.square_to_index(dest_square)

        move_tensor = torch.zeros(64, 64)
        move_tensor[source_index][dest_index] = 1

        return move_tensor

    def square_to_index(self, square):
        rank = 8 - int(square[1])  # Ranks numbered 8 to 1
        file = ord(square[0]) - ord('a')  # Files lettered a to h
        return rank * 8 + file

    def output_to_move(self, output):
        output_matrix = output.view(64, 64)

        # Find indices of maximum val in the output matrix
        source_index, dest_index = torch.unravel_index(torch.argmax(output_matrix), output_matrix.shape)

        # Convert the indices to a move in algebraic notation
        move = self.index_to_square(source_index.item()) + self.index_to_square(dest_index.item())

        return move


    def modify_output(self, output):
        # Add a small random noise to the output tensor
        noise = torch.randn_like(output) * 0.01
        output += noise

        return output

    def index_to_square(self, index):
        # Convert the index to a rank and file
        rank = 8 - (index // 8)  # Ranks are numbered from 8 to 1
        file = chr((index % 8) + ord('a'))  # Files are lettered from a to h

        # Return the square in algebraic notation
        return file + str(rank)

    def fen_to_tensor(self, fen):
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(8,8))#, stride=1, padding=1) # kernel size of (8,8)
        self.fc1 = nn.Linear(64 + 2, 1024)  # 64 input nodes + 1 input node for turn + 1 node for check
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 64*64)  # 64*64 output nodes

    def forward(self, x, scalar_inputs):
        x = x.view(x.size(0), 1, 8, 8)
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

    # Create a chess board from the FEN string
    board = chess.Board(fen)

    # Convert the move to a chess.Move object
    move = chess.Move.from_uci(move)

    # Check if the move is legal
    return move in board.legal_moves


def is_check(fen):
    board = chess.Board(fen)
    return board.is_check()


def custom_loss(dataset, output, target, fen, illegal_move_penalty=5000.0):
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
        move = dataset.output_to_move(output[i])
        if not is_legal_move(fen[i], move):
            penalties[i] = illegal_move_penalty

    loss += penalties.mean()

    return loss


def train(model, dataset, data_loader, criterion, optimizer, num_epochs):
    print('Begin Training!')
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        try:
            for i, data in enumerate(data_loader):

                turns = data['white_to_move'].to(device)
                checks = data['is_check'].to(device)

                inputs = data['fen'].to(device)
                labels = data['move'].to(device)
                fen = data['fen_str']

                scalar_inputs = torch.cat((turns.unsqueeze(0), checks.unsqueeze(0)), dim=0).T

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs, scalar_inputs)
                loss = criterion(dataset, outputs, labels, fen)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        except:
            print("Finished Training Early!")
            break
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(data_loader)}')

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

    fen_tensor = ChessIterableDataset(None, None).fen_to_tensor(fen).to(device)

    move_tensor = torch.tensor(white_to_move).unsqueeze(0).to(device)
    check_tensor = torch.tensor(is_check(fen)).unsqueeze(0).to(device)

    scalar_inputs = torch.cat((move_tensor, check_tensor), dim=0).T

    # input_all = torch.cat((fen_tensor, move_tensor, check_tensor), dim=0).to(device)

    # Add an extra dimension to the tensor and move it to the same device as the model
    # input = input_all.to(next(model.parameters()).device)

    # print(input.shape)

    # Set the model to evaluation mode
    model.eval()

    # Get the model's predictions
    with torch.no_grad():
        output = model(fen_tensor.unsqueeze(0), scalar_inputs.unsqueeze(0))

    output = output.cpu() # Move tensor back to cpu

    # Convert the output tensor to a move
    move = ChessIterableDataset(None, None).output_to_move(output[0])

    # If the move is illegal, modify the output tensor to suggest a different move
    while not is_legal_move(fen, move):
        print(f'UserWarning: Illegal move {move}, modifying output tensor...')
        
        # Modify output tensor
        # output = modify_output(output)
        # move = ChessIterableDataset(None, None).output_to_move(output[0])

        # generate from a list of the legal moves
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        random_move = np.random.choice(legal_moves)
        move = random_move.uci()

    return move




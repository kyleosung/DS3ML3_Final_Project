import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

import chess

import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import IterableDataset #, DataLoader, Dataset

from tqdm import tqdm

torch.set_default_dtype(torch.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_NUMBER = 8

class EvalNet(nn.Module):
    """
    Neural network model for evaluating chess positions.

    This model takes a chess position as input and predicts the evaluation score
    for that position. It consists of convolutional and fully connected layers.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.

    Methods:
        forward(x): Performs forward pass through the network.
    """

    def __init__(self):
        """
        Initializes the EvalNet class

        Args:
            None

        Returns:
            None
        """
        super(EvalNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size = 5, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(16, 24, kernel_size = 3, stride = 1, padding = 1) 
        self.fc1 = nn.Linear(24 * 6 * 6 + 10, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x, piece_counts=None):
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 6, 8, 8)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.cat((x, piece_counts), dim=1)
        x = F.leaky_relu(self.fc1(x))
        return self.fc2(x)   

class ChessIterableDataset(IterableDataset):

    def __init__(self, csv_files, chunksize=45000):
        self.csv_files = csv_files
        self.chunksize = chunksize

    def process_chunk(self, chunk):
        """
        Process a chunk of data and return a list of samples.

        Args:
            chunk (pandas.DataFrame): The chunk of data to process.

        Returns:
            list: A list of samples, where each sample is a dictionary containing the following keys:
                - 'fen': The board tensor representation.
                - 'fen_str': The FEN string representation of the board.
                - 'white_active': The indicator of whether it is white's turn to move.
                - 'cp': The centipawn evaluation of the board position.
        """
        # Process the chunk and return a list of samples
        samples = []
        for index, row in chunk.iterrows():
            
            # Feature Variables
            board = row['board']
            white_active = row['white_active']
            cp = row['cp']
            
            if '#' in str(cp) and white_active:
                cp = 10
            elif '#' in str(cp) and not white_active:
                cp = -10
            elif cp > 9:
                cp = 9
            elif cp < -9:
                cp = -9
                

            # Convert data to tensors
            board_tensor = fen_str_to_3d_tensor(board)
            white_active = torch.tensor(white_active, dtype=torch.float32)
            
            num_pieces = get_number_of_pieces(board)

            cp = torch.tensor(cp, dtype=torch.float32)
        
            samples.append({'fen': board_tensor, 
                            'fen_str': board, 
                            'white_active': white_active, 
                            'cp': cp, 
                            'num_pieces': num_pieces,
            })
        return samples


    def __len__(self):
        return sum(1 for _ in self.__iter__())


    def __getitem__(self, idx):
        board = self.dataframe.iloc[idx]['board']
        white_active = self.dataframe.iloc[idx]['white_active']
        cp = self.dataframe.iloc[idx]['cp']
        
        # Convert data to tensors
        board_tensor = fen_str_to_3d_tensor(board)
        white_active = torch.tensor(white_active, dtype=torch.float32)

        num_pieces = get_number_of_pieces(board)

        if '#' in str(cp) and white_active:
            cp = 10
        elif '#' in str(cp) and not white_active:
            cp = -10
        elif cp > 9:
            cp = 9
        elif cp < -9:
            cp = -9

        cp = torch.tensor(cp, dtype=torch.float32)

        return {'fen': board_tensor, 'fen_str': board, 'white_active': white_active, 'cp': cp, 'num_pieces': num_pieces}
        

    def __iter__(self):
        for idx, csv_file in enumerate(self.csv_files):
            # Add usecols 2024-04-09. Hopefully should make it load faster.
            chunk_iter = pd.read_csv(csv_file, chunksize=self.chunksize, 
                                     usecols=['board', 'cp', 'white_active'],#, 'white_elo', 'black_elo'], 
                                     dtype = {'board': 'object',
                                              'white_active': 'bool', 
                                              'cp': 'object'
                                            #   'white_elo': 'uint16', 
                                            #   'black_elo': 'uint16',
                                    },
                                     # low_memory = False,
            )

            if idx % 10 == 0:
                # print(f'Read file {idx} of {len(self.csv_files)}')
                pass

            for chunk in chunk_iter:
                chunk = chunk.dropna(how='any')
                # chunk = chunk.loc[chunk['white_elo'] >= 1000]
                # chunk = chunk.loc[chunk['black_elo'] >= 1000]
                
                chunk['cp'] = pd.to_numeric(chunk['cp'], errors='coerce')

                chunk.loc[(chunk['cp'].isna() | chunk['cp'].isnull() ) & (chunk.loc[chunk['cp'].isna(), 'white_active'] == 1), 'cp'] = 9
                chunk.loc[(chunk['cp'].isna() | chunk['cp'].isnull() ) & (chunk.loc[chunk['cp'].isna(), 'white_active'] == 0), 'cp'] = -9

                chunk.loc[(chunk['cp'] > 9), 'cp'] = 9
                chunk.loc[(chunk['cp'] < -9), 'cp'] = -9

                yield from self.process_chunk(chunk) # Returns a list of yielded elements
            
            del chunk_iter # clean up garbage




class ChessIterableDataset_Large(IterableDataset):

    def __init__(self, csv_files):
        self.csv_files = csv_files

    def process_chunk(self, chunk):
        """
        Process a chunk of data and return a list of samples.

        Args:
            chunk (pandas.DataFrame): The chunk of data to process.

        Returns:
            list: A list of samples, where each sample is a dictionary containing the following keys:
                - 'fen': The board tensor representation.
                - 'fen_str': The FEN string representation of the board.
                - 'white_active': The indicator of whether it is white's turn to move.
                - 'cp': The centipawn evaluation of the board position.
        """
        # Process the chunk and return a list of samples
        samples = []
        for index, row in chunk.iterrows():
            
            # Feature Variables
            board = row['board']
            white_active = row['white_active']
            cp = row['cp']
            
            if '#' in str(cp) and white_active:
                cp = 10
            elif '#' in str(cp) and not white_active:
                cp = -10
            elif cp > 9:
                cp = 9
            elif cp < -9:
                cp = -9
                

            # Convert data to tensors
            board_tensor = fen_str_to_3d_tensor(board)
            white_active = torch.tensor(white_active, dtype=torch.float32)
            
            num_pieces = get_number_of_pieces(board)

            cp = torch.tensor(cp, dtype=torch.float32)
        
            samples.append({'fen': board_tensor, 
                            'fen_str': board, 
                            'white_active': white_active, 
                            'cp': cp, 
                            'num_pieces': num_pieces,
            })
        return samples


    def __len__(self):
        return sum(1 for _ in self.__iter__())


    def __getitem__(self, idx):
        """
        Retrieves item at specified index in dataset

        Note to self: this method is part of the implementation of the PyTorch Dataset interface. It is used by the DataLoader to fetch individual samples from the dataset.

        Parameters
        ----------
        idx : int
            index of the item to retrieve.

        Returns
        -------
        dict
            Contains the board state, the active player, and the centipawn value. 
            The board state is a 3D tensor, the active player is a float tensor (1.0 for white, 0.0 for black), and the centipawn value is a float.

        """
        board = self.dataframe.iloc[idx]['board']
        white_active = self.dataframe.iloc[idx]['white_active']
        cp = self.dataframe.iloc[idx]['cp']
        
        # Convert data to tensors
        board_tensor = fen_str_to_3d_tensor(board)
        white_active = torch.tensor(white_active, dtype=torch.float32)

        num_pieces = get_number_of_pieces(board)

        if '#' in str(cp) and white_active:
            cp = 10
        elif '#' in str(cp) and not white_active:
            cp = -10
        elif cp > 9:
            cp = 9
        elif cp < -9:
            cp = -9

        cp = torch.tensor(cp, dtype=torch.float32)

        return {'fen': board_tensor, 'fen_str': board, 'white_active': white_active, 'cp': cp, 'num_pieces': num_pieces}
        

    def __iter__(self):
        for csv_file in self.csv_files:
            # Add usecols 2024-04-09. Hopefully should make it load faster.
            dataframe = pd.read_csv(csv_file, 
                                    usecols = ['board', 'cp', 'white_active'],#, 'white_elo', 'black_elo'], 
                                    dtype = {'board': 'object',
                                             'white_active': 'bool',
                                             'cp': 'object',
                                            #  'white_elo': 'uint16', 
                                            #  'black_elo': 'uint16', 
                                    },
                                    # low_memory = False,
            )
            
            dataframe = dataframe.dropna(how = 'any')
            # dataframe = dataframe.loc[dataframe['white_elo'] >= 1000]
            # dataframe = dataframe.loc[dataframe['black_elo'] >= 1000]
            
            dataframe['cp'] = pd.to_numeric(dataframe['cp'], errors='coerce')

            dataframe.loc[(dataframe['cp'].isna() | dataframe['cp'].isnull() ) & (dataframe.loc[dataframe['cp'].isna(), 'white_active'] == 1), 'cp'] = 9
            dataframe.loc[(dataframe['cp'].isna() | dataframe['cp'].isnull() ) & (dataframe.loc[dataframe['cp'].isna(), 'white_active'] == 0), 'cp'] = -9

            dataframe.loc[(dataframe['cp'] > 9), 'cp'] = 9
            dataframe.loc[(dataframe['cp'] < -9), 'cp'] = -9

            yield from self.process_chunk(dataframe) # Returns a list of yielded elements




# def fen_str_to_1d_array(fen):
#     """
#     Converts a FEN string representation of a chess board to a 1-d vector array representation.

#     Args:
#         fen (str): The FEN string representing the chess board.

#     Returns:
#         np.ndarray: A array vector representation of the chess board.

#     Example:
#         >>> fen_str_to_flat_tensor('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
#         tensor([[ -4.,  -2.,  -3.,  -5.,  -6.,  -3.,  -2.,  -4.],
#                 [ -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.],
#                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
#                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
#                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
#                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
#                 [  1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.],
#                 [  4.,   2.,   3.,   5.,   6.,   3.,   2.,   4.]])
#     """    
#     # Define a mapping from pieces to integers
#     piece_to_int = {
#         'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
#         'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,
#     }

#     # Split the FEN string into parts ## 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
#     parts = fen.split(' ')
#     ranks = parts[0].split('/') # Only process the board position (the first part)

#     # Convert the ranks to a list of integers
#     board = []
#     for rank in ranks:
#         for char in rank:
#             if char.isdigit():
#                 # If the character is a digit, add that many zeros to the board
#                 board.extend([0] * int(char))
#             else:
#                 # Otherwise, add the integer representation of the piece to the board
#                 board.append(piece_to_int[char])

#     # Convert the board to a tensor
#     board_array = np.array(board, dtype='float32')

#     return board_array

# def fen_str_to_flat_tensor(fen):
#     """
#     Converts a FEN string representation of a chess board to a flat tensor representation.

#     Args:
#         fen (str): The FEN string representing the chess board.

#     Returns:
#         torch.Tensor: A flat tensor representation of the chess board.

#     Example:
#         >>> fen_str_to_flat_tensor('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
#         tensor([[ -4.,  -2.,  -3.,  -5.,  -6.,  -3.,  -2.,  -4.],
#                 [ -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.],
#                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
#                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
#                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
#                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
#                 [  1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.],
#                 [  4.,   2.,   3.,   5.,   6.,   3.,   2.,   4.]])
#     """    
#     # Define a mapping from pieces to integers
#     piece_to_int = {
#         'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
#         'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,
#     }

#     # Split the FEN string into parts ## 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
#     parts = fen.split(' ')
#     ranks = parts[0].split('/') # Only process the board position (the first part)

#     # Convert the ranks to a list of integers
#     board = []
#     for rank in ranks:
#         for char in rank:
#             if char.isdigit():
#                 # If the character is a digit, add that many zeros to the board
#                 board.extend([0] * int(char))
#             else:
#                 # Otherwise, add the integer representation of the piece to the board
#                 board.append(piece_to_int[char])

#     # Convert the board to a tensor
#     board_tensor = torch.tensor(board, dtype=torch.float32).reshape(8,8)

#     return board_tensor



def fen_str_to_3d_tensor(fen):
    """
    Converts a FEN string representation of a chess position to a 3D tensor.

    Args:
        fen (str): The FEN string representing the chess position.

    Returns:
        torch.Tensor: A 3D tensor representing the chess position, where each element
                      corresponds to a piece on the board.

    Example:
        fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        tensor = fen_str_to_3d_tensor(fen)
    """
    piece_to_int = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,
    }

    board = np.zeros((6, 8, 8), dtype=np.float32)
    
    # Split the FEN string into parts ## 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    fen_parts = fen.split(' ')
    fen_rows = fen_parts[0].split('/') # Only process the board position (the first part)
    
    for row_idx, row in enumerate(fen_rows):
        col_idx = 0
        for char in row:
            if char.isdigit():
                col_idx += int(char)
            else:
                piece = piece_to_int[char]
                board[abs(piece) - 1, row_idx, col_idx] = piece
                col_idx += 1
    
    return torch.tensor(board)



def train(model, train_data_loader, val_data_loader, criterion, optimizer, num_epochs):
    """
    Trains model

    Parameters
    ----------
    model : torch.nn.Module
        model to be trained.
    train_data_loader : torch.utils.data.DataLoader
        training data.
    val_data_loader : torch.utils.data.DataLoader
        validation data.
    criterion : torch.nn.modules.loss._Loss
        loss function
    optimizer : torch.optim.Optimizer
        optimizer
    num_epochs : int
        Number of epochs

    Returns
    -------
    list
        average training loss for each epoch
    list
        average validation loss for each epoch

    """
    print(f'Begin Training! (on {device})')

    training_loss_history = []
    validation_loss_history = []

    try:
        for epoch in tqdm(range(num_epochs)):

            train_running_loss = 0.0
            val_running_loss = 0.0

            ## TRAINING PHASE =================================
            model.train()  # Set the model to training mode

            for i, train_data in enumerate(train_data_loader):

                # Feature Variables
                fen = train_data['fen'].to(device)
                # white_active = train_data['white_active'].to(device).unsqueeze(0)

                # Predictor Variables
                cp = ((train_data['cp']).to(device)).unsqueeze(1)

                pieces_tensor = train_data['num_pieces'].to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                train_outputs = model(fen, pieces_tensor)

                train_batch_loss = criterion(train_outputs, cp)

                train_batch_loss.backward()
                optimizer.step()

                train_running_loss += train_batch_loss.item()
            
            ## VALIDATION PHASE =================================
            model.eval()  # Set the model to evaluation mode
        
            with torch.no_grad():
                for i, val_data in enumerate(val_data_loader):
                    # Extract the inputs and labels from the validation data
                    
                    # Feature Variables
                    fen = val_data['fen'].to(device)
                    # white_active  = val_data['white_active'].to(device).unsqueeze(0)

                    pieces_tensor = val_data['num_pieces'].to(device)

                    # Predictor Variables
                    cp = (val_data['cp'].to(device)).unsqueeze(1)
                    
                    # Forward pass
                    val_outputs = model(fen, pieces_tensor)
                    val_batch_loss = criterion(val_outputs, cp)

                    val_running_loss += val_batch_loss.item()

            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_running_loss/len(train_data_loader):.5f}, Validation Loss: {val_running_loss/len(val_data_loader):.5f}')
            training_loss_history.append(train_running_loss/len(train_data_loader))
            validation_loss_history.append(val_running_loss/len(val_data_loader))
            
    except KeyboardInterrupt:
        print("Manual Stop: Finished Training Early!")
    finally:
        torch.save(model, f'models_autosave/autosave{MODEL_NUMBER}-{1}.pth')

    print(f'Finished Training!')

    

    return training_loss_history, validation_loss_history



def get_number_of_pieces(fen_str):
    """
    Get the number of pieces of a given type and color on the board.

    Parameters
    ----------
    board : chess.Board
        The chess board.

    Returns
    -------
    list of int, length 10
        Number of pieces on the board (not including kings)
        [white_pawns, white_knights, ... , black_rooks, black_queens]
    """
    
    piece_counts = []
    max_starting_pieces = [8, 2, 2, 2, 1]
    values = [1, 3, 3.1, 5, 9]

    board = chess.Board(fen_str)

    for j, color in enumerate([chess.WHITE, chess.BLACK]): # 0 for white, 1 for black
        for i, piece in enumerate([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]):
            piece_counts.append(len(board.pieces(piece, color)) / max_starting_pieces[i] * values[i] * (-1)**j) # White pieces are positive, black pieces are negative
    
    return torch.tensor(piece_counts).to(device)




def predict(model, fen, move_number=0, stochastic=True):
    """
    Predicts the evaluation of all legal moves in a given chess position.

    Parameters
    ----------
    model : torch.nn.Module
        trained PyTorch model used for prediction
    fen : str
        FEN string representing the chess position.
    move_number : int, optional
        The number of the move in the game. Default: 0.
    stochastic : bool, optional
        If True, predictions are stochastic (only in the first seven moves)
        If False, predictions are deterministic

    Returns
    -------
    chess.Move object representing the best move to play.
    """
    board = chess.Board(fen)
    legal_moves_list = list(board.legal_moves)
    evals_list = []

    model.eval()
    with torch.no_grad():
        for move in legal_moves_list:

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # is_capture = board.is_capture(move)

            board.push(move)
            fen_tensor = fen_str_to_3d_tensor(board.fen()).unsqueeze(0).to(device)
            # print(fen_tensor.shape)

            pieces_counts = get_number_of_pieces(board.fen()).unsqueeze(0).to(device)

            evals_list.append(float(model(fen_tensor, pieces_counts).to('cpu')))

            if board.is_checkmate():
                return move # Always make a move which gives checkmate if possible.

            board.pop()

            # New portion (added 2024-04-09)
            if board.is_capture(move):
                if board.turn:
                    evals_list[-1] += 0.5 # Modify to add piece value eventually
                else:
                    evals_list[-1] -= 0.5 # Modify to add piece value eventually
    

    evals_list = np.array(evals_list)
    # print(evals_list)
    # print(np.array(legal_moves_list))

    sorted_indices = np.argsort(evals_list)
    
    # print(sorted_indices)

    if board.turn:
        '''
        if it's white's turn, we must reverse the array such that the highest evaluation is first
        if it's black's turn, keep the array ascending such that the lowest evaluation for the white pieces is first
        ''' 
        sorted_indices = sorted_indices[::-1]
    
    # print(np.array(legal_moves_list).shape)

    # Use the sorted indices to sort legal_moves and evals_list
    sorted_legal_moves = np.array(legal_moves_list)[sorted_indices]
    sorted_evals_list = evals_list[sorted_indices]

    if not stochastic: # if not using stochastic mode return best move
        return sorted_legal_moves[0]

    sample = np.random.random_sample()

    # print(sample)
    # print(sorted_legal_moves)

    if sample <= 0.65 or move_number > 7: # 65% chance for best move
        # print(f'playing best move')
        return sorted_legal_moves[0]
    elif sample <= 0.85 or move_number > 5: # 25% chance for second-best move
        return sorted_legal_moves[1]
    elif sample <= 0.975 or move_number > 3: #  7.5% chance for third-best move
        return sorted_legal_moves[2]
    else: # 2.5% chance for fourth-best move
        return sorted_legal_moves[3]
    

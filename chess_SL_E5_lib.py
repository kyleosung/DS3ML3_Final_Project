import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import chess

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, Dataset

torch.set_default_dtype(torch.float32)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EvalNet(nn.Module):
    def __init__(self):
        super(EvalNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size = 5, stride = 1, padding = 1)
        self.fc1 = nn.Linear(576, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        return self.fc2(x) # no activation in output layer
        
    



class ChessIterableDataset(IterableDataset):

    def __init__(self, csv_files, chunksize):
        self.csv_files = csv_files
        self.chunksize = chunksize

    def process_chunk(self, chunk):
        # Process the chunk and return a list of samples
        samples = []
        for index, row in chunk.iterrows():
            
            # Feature Variables
            board = row['board']
            white_active = row['white_active']
            cp = row['cp']
            
            if '#' in str(cp) and white_active:
                cp = 15
            elif '#' in str(cp) and not white_active:
                cp = -15
            elif cp > 14:
                cp = 14
            elif cp < -14:
                cp = -14
                

            # Convert data to tensors
            board_tensor = fen_str_to_3d_tensor(board)
            white_active = torch.tensor(white_active, dtype=torch.float32)
            
            cp = torch.tensor(cp, dtype=torch.float32)
        
            samples.append({'fen': board_tensor, 
                            'fen_str': board, 
                            'white_active': white_active, 
                            'cp': cp, 
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

        if '#' in str(cp) and white_active:
            cp = 15
        elif '#' in str(cp) and not white_active:
            cp = -15
        elif cp > 14:
            cp = 14
        elif cp < -14:
            cp = -14

        cp = torch.tensor(cp, dtype=torch.float32)

        return {'fen': board_tensor, 'fen_str': board, 'white_active': white_active, 'cp': cp}
        

    def __iter__(self):
        for idx, csv_file in enumerate(self.csv_files):
            chunk_iter = pd.read_csv(csv_file, chunksize=self.chunksize)
            
            if idx % 10 == 0:
                # print(f'Read file {idx} of {len(self.csv_files)}')
                pass

            for chunk in chunk_iter:
                chunk = chunk.dropna(how='any')
                chunk = chunk.loc[chunk['white_elo'] >= 1400]
                chunk = chunk.loc[chunk['black_elo'] >= 1400]
                
                chunk['cp'] = pd.to_numeric(chunk['cp'], errors='coerce')

                chunk.loc[(chunk['cp'].isna() | chunk['cp'].isnull() ) & (chunk.loc[chunk['cp'].isna(), 'white_active'] == 1), 'cp'] = 15
                chunk.loc[(chunk['cp'].isna() | chunk['cp'].isnull() ) & (chunk.loc[chunk['cp'].isna(), 'white_active'] == 0), 'cp'] = -15

                chunk.loc[(chunk['cp'] > 15), 'cp'] = 15
                chunk.loc[(chunk['cp'] < -15), 'cp'] = -15

                yield from self.process_chunk(chunk) # Returns a list of yielded elements
            
            del chunk_iter # clean up garbage





def fen_str_to_flat_tensor(fen):
    # Define a mapping from pieces to integers
    piece_to_int = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,
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
    board_tensor = torch.tensor(board, dtype=torch.float32).reshape(8,8)

    return board_tensor



def fen_str_to_3d_tensor(fen):
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
    print('Begin Training!')

    training_loss_history = []
    validation_loss_history = []

    try:
        for epoch in range(num_epochs):

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

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                train_outputs = model(fen)

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

                    # Predictor Variables
                    cp = (val_data['cp'].to(device)).unsqueeze(1)
                    
                    # Forward pass
                    val_outputs = model(fen)
                    val_batch_loss = criterion(val_outputs, cp)

                    val_running_loss += val_batch_loss.item()

            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_running_loss/len(train_data_loader):.5f}, Validation Loss: {val_running_loss/len(val_data_loader):.5f}')
            training_loss_history.append(train_running_loss/len(train_data_loader))
            validation_loss_history.append(val_running_loss/len(val_data_loader))
            
    except KeyboardInterrupt:
        print("Manual Stop: Finished Training Early!")

    print('Finished Training!')

    torch.save(model, 'models_autosave/autosave5.pth')

    return training_loss_history, validation_loss_history



def predict(model, fen, stochastic=True):
    board = chess.Board(fen)
    legal_moves_list = np.array(board.legal_moves)
    evals_list = []

    model.eval()
    with torch.no_grad():
        for move in legal_moves_list:

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            is_capture = board.is_capture(move)

            board.push(move)
            fen_tensor = fen_str_to_3d_tensor(board.fen()).unsqueeze(0).to(device)
            
            evals_list.append(model(fen_tensor).to('cpu'))
            board.pop()

    evals_list = np.array(evals_list)
    sorted_indices = np.argsort(evals_list)
    
    if board.turn:
        '''
        if it's white's turn, we must reverse the array such that the highest evaluation is first
        if it's black's turn, keep the array ascending such that the lowest evaluation for the white pieces is first
        ''' 
        sorted_indices = sorted_indices[::-1]
    
    # Use the sorted indices to sort legal_moves and evals_list
    sorted_legal_moves = legal_moves_list[sorted_indices]
    sorted_evals_list = evals_list[sorted_indices]

    if not stochastic: # if not using stochastic mode return best move
        return sorted_legal_moves[0]

    sample = np.random.random_sample()

    if sample <= 0.65: # 65% chance for best move
        return sorted_legal_moves[0]
    elif sample <= 0.85: # 20% chance for second-best move
        return sorted_legal_moves[1]
    elif sample <= 0.95: #  10% chance for third-best move
        return sorted_legal_moves[2]
    else: # 5% chance for fourth-best move
        return sorted_legal_moves[3]
    
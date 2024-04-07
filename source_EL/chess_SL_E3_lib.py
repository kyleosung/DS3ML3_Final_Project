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
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 5, stride = 1, padding = 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(576 + 2, 512) ## Add two for scalar inputs
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x, scalar_inputs, train=True):
        # print(x.shape)
        x = F.leaky_relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        if not train:
            x = x.view(1, -1)

        # print(f"x shape: {x.shape}, scalar_input shape: {scalar_inputs.shape}")
        x = torch.cat((x, scalar_inputs), dim=1)
        # print(x.shape)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x
    



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
            white_active = row['white_active']
            cp = row['cp']
            is_check = row['is_check'] # add to predictions

            # print(cp, type(cp))

            if '#' in str(cp) and white_active:
                cp = 15
            elif '#' in str(cp) and not white_active:
                cp = -15
            elif cp > 15:
                cp = 15
            elif cp < -15:
                cp = -15
                

            # Convert data to tensors
            board_tensor = fen_str_to_tensor(board)
            white_active = torch.tensor(white_active, dtype=torch.float32)
            
            cp = torch.tensor(cp, dtype=torch.float32)
            is_check = torch.tensor(is_check, dtype=torch.float32)
        
            samples.append({'fen': board_tensor, 'fen_str': board, 'white_active': white_active, 'cp': cp, 'is_check': is_check})
        return samples


    def __len__(self):
        return sum(1 for _ in self.__iter__())


    def __getitem__(self, idx):
        board = self.dataframe.iloc[idx]['board']
        white_active = self.dataframe.iloc[idx]['white_active']
        cp = self.dataframe.iloc[idx]['cp']
        is_check = self.dataframe.iloc[idx]['is_check']
        
        # Convert data to tensors
        board_tensor = fen_str_to_tensor(board)
        white_active = torch.tensor(white_active, dtype=torch.float32)

        if '#' in str(cp) and white_active:
            cp = 15
        elif '#' in str(cp) and not white_active:
            cp = -15
        elif cp > 15:
            cp = 15
        elif cp < -15:
            cp = -15

        cp = torch.tensor(cp, dtype=torch.float32)

        return {'fen': board_tensor, 'fen_str': board, 'white_active': white_active, 'cp': cp, 'is_check': is_check}
    

    def __iter__(self):
        for idx, csv_file in enumerate(self.csv_files):
            chunk_iter = pd.read_csv(csv_file, chunksize=self.chunksize)#, dtype={'cp': float})#, dtype={15:str, 33:str})#, engine='pyarrow')
            
            if idx % 10 == 0:
                # print(f'Read file {idx} of {len(self.csv_files)}')
                pass

            for chunk in chunk_iter:
                chunk = chunk.dropna(how='any')
                chunk = chunk.loc[chunk['white_elo'] >= 1350] # select a subset of players with a certain rating
                chunk = chunk.loc[chunk['black_elo'] >= 1350]
                
                chunk['cp'] = pd.to_numeric(chunk['cp'], errors='coerce')

                chunk.loc[(chunk['cp'].isna() | chunk['cp'].isnull() ) & (chunk.loc[chunk['cp'].isna(), 'white_active'] == 1), 'cp'] = 15
                chunk.loc[(chunk['cp'].isna() | chunk['cp'].isnull() ) & (chunk.loc[chunk['cp'].isna(), 'white_active'] == 0), 'cp'] = -15

                chunk.loc[(chunk['cp'] > 15), 'cp'] = 15
                chunk.loc[(chunk['cp'] < -15), 'cp'] = -15

                yield from self.process_chunk(chunk) # Returns a list of yielded elements
            
            del chunk_iter # clean up garbage





def fen_str_to_tensor(fen):
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
    board_tensor = torch.tensor(board, dtype=torch.float32).reshape(8,8)

    return board_tensor



def train(model, train_data_loader, val_data_loader, criterion, optimizer, num_epochs):
    print('Begin Training!')
    model.train()  # Set the model to training mode

    training_loss_history = []
    validation_loss_history = []

    for epoch in range(num_epochs):
        train_running_loss = 0.0
        val_running_loss = 0.0
        try:
            ## TRAINING PHASE
            for i, data in enumerate(train_data_loader):

                # Feature Variables
                fen           = data['fen'].to(device).unsqueeze(1)
                white_active  = data['white_active'].to(device).unsqueeze(0)
                is_check      = data['is_check'].to(device).unsqueeze(0)
                scalar_inputs = torch.cat( (white_active, is_check), dim = 0 ).T

                # Predictor Variables
                cp = ((data['cp']).to(device)).unsqueeze(1)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                train_outputs = model(fen, scalar_inputs)

                # print(train_outputs.shape, cp.shape)

                # print(torch.isnan(train_outputs).sum(), torch.isnan(cp).sum())

                train_batch_loss = criterion(train_outputs, cp)

                # print(train_batch_loss)

                # Backward pass and optimization
                train_batch_loss.backward()
                optimizer.step()

                train_running_loss += train_batch_loss.item()
            
            ## VALIDATION PHASE
            model.eval()  # Set the model to evaluation mode
        
            with torch.no_grad():
                for i, val_data in enumerate(val_data_loader):
                    # Extract the inputs and labels from the validation data
                    
                    # Feature Variables
                    fen           = val_data['fen'].to(device).unsqueeze(1)
                    white_active  = val_data['white_active'].to(device).unsqueeze(0)
                    is_check      = val_data['is_check'].to(device).unsqueeze(0)
                    scalar_inputs = torch.cat( (white_active, is_check), dim = 0 ).T

                    # Predictor Variables
                    cp = (val_data['cp'].to(device)).unsqueeze(1)
                    
                    # Forward pass
                    val_outputs = model(fen, scalar_inputs)
                    val_batch_loss = criterion(val_outputs, cp)
                    
                    # print(val_batch_loss)

                    val_running_loss += val_batch_loss.item()

        except KeyboardInterrupt:
            print("Manual Stop: Finished Training Early!")
            break
        
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_running_loss/len(train_data_loader):.5f}, Validation Loss: {val_running_loss/len(val_data_loader):.5f}')
        training_loss_history.append(train_running_loss/len(train_data_loader))
        validation_loss_history.append(val_running_loss/len(val_data_loader))

    print('Finished Training!')

    torch.save(model, 'models/autosave.pth')

    return training_loss_history, validation_loss_history



def predict(model, fen):
    board = chess.Board(fen)
    legal_moves_list = list(board.legal_moves)
    evals_list = []

    model.eval()
    with torch.no_grad():
        for move in legal_moves_list:

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            board.push(move)
            fen_tensor = fen_str_to_tensor(board.fen()).unsqueeze(0).to(device)

            # print(fen_tensor.shape)

            white_active = torch.tensor(board.turn, dtype=torch.float32)
            is_check = torch.tensor(board.is_check(), dtype=torch.float32)
            scalar_inputs = torch.vstack( (white_active, is_check)).T.to(device)
            # print(scalar_inputs.shape)

            
            evals_list.append(model(fen_tensor, scalar_inputs, False).to('cpu'))
            board.pop()

    if board.turn:
        return legal_moves_list[np.argmax(evals_list)].uci()
    else:
        return legal_moves_list[np.argmin(evals_list)].uci() 
    
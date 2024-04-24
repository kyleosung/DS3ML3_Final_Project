# Standard Utilities
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Sklearn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR

# Chess
import chess

# File Navigation
import glob
from pathlib import Path
import pickle
import joblib

# Misc utilities
from tqdm import tqdm
import time





def fen_str_to_1d_array(fen):
    """
    Converts a FEN string representation of a chess board to a flat tensor representation.

    Args:
        fen (str): The FEN string representing the chess board.

    Returns:
        torch.Tensor: A flat tensor representation of the chess board.

    Example:
        >>> fen_str_to_flat_tensor('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        tensor([[ -4.,  -2.,  -3.,  -5.,  -6.,  -3.,  -2.,  -4.],
                [ -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.],
                [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                [  1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.],
                [  4.,   2.,   3.,   5.,   6.,   3.,   2.,   4.]])
    """    
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
    board_array = np.array(board, dtype='float32')

    return board_array





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
            piece_counts.append(len(board.pieces(piece, color)) / max_starting_pieces[i] * values[i] * (-1)**j / 2) # White pieces are positive, black pieces are negative
    
    return np.array(piece_counts)






def preprocess_XY(fen_str_all, cp, white_active_all):
    X = []
    y = cp
    
    for i in tqdm(range(len(y))):

        fen_str = fen_str_all[i]

        if np.isnan(y[i]) and white_active_all[i]:
            y[i] = 10
        elif np.isnan(cp[i]) and not white_active_all[i]:
            y[i] = -10
        elif y[i] > 9:
            y[i] = 9
        elif y[i] < -9:
            y[i] = -9

        piece_counts = get_number_of_pieces(fen_str)

        inputs = np.concatenate((fen_str_to_1d_array(fen_str), piece_counts))

        X.append(inputs)

    X = np.array(X, dtype = 'float32')
    y = np.array(y, dtype = 'float32')

    return X, y





def load_data_XY_small(path = './data_SK', test_size = 0.2):
    X_loaded = joblib.load(f'{path}/data_X.joblib')
    y_loaded = joblib.load(f'{path}/data_y.joblib')

    return train_test_split(X_loaded, y_loaded, random_state=0, test_size=test_size)



def load_data_XY_large(path = './data_SK', test_size = 0.2):
    X_loaded = joblib.load(f'{path}/data_X_large.joblib')
    y_loaded = joblib.load(f'{path}/data_y_large.joblib')

    return train_test_split(X_loaded, y_loaded, random_state=0, test_size=test_size)


def load_data_XY_a(path = './data_SK', test_size = 0.2):
    X_loaded = joblib.load(f'{path}/data_X_a.joblib')
    y_loaded = joblib.load(f'{path}/data_y_a.joblib')

    return train_test_split(X_loaded, y_loaded, random_state=0, test_size=test_size)


def load_data_XY_a_to_b(path = './data_SK', test_size = 0.2):
    X_loaded = joblib.load(f'{path}/data_X_a-b.joblib')
    y_loaded = joblib.load(f'{path}/data_y_a-b.joblib')

    return train_test_split(X_loaded, y_loaded, random_state=0, test_size=test_size)


def load_data_XY_a_to_d(path = './data_SK', test_size = 0.2):
    X_loaded = joblib.load(f'{path}/data_X_a-d.joblib')
    y_loaded = joblib.load(f'{path}/data_y_a-d.joblib')

    return train_test_split(X_loaded, y_loaded, random_state=0, test_size=test_size)


def get_current_version(model_type, path = './joblib'):
    '''
    Model_type: KNN, LR, RF, SVR
    '''

    model_file_path = f'{path}/model_{model_type}_1.joblib'
    counter = 1

    while Path(model_file_path).is_file(): # ensure that no files are overwritten
        counter += 1
        model_file_path = f'{path}/model_{model_type}_{counter}.joblib'
    
    return counter




def MAD(y_test, y_pred):
    return np.sum(np.abs(y_pred - y_test)) / len(y_test)






def predict_SK(model, fen, move_number = 5, stochastic = True):
    '''
    ### NOTE TO SELF: model should be an sklearn regressor with a predict method
    '''

    board = chess.Board(fen)
    legal_moves_list = list(board.legal_moves)
    evals_list = []
    
    for move in legal_moves_list:

        board.push(move)
        fen_array = fen_str_to_1d_array(board.fen())

        pieces_counts = get_number_of_pieces(board.fen())

        inputs = np.concatenate((fen_array, pieces_counts))
        inputs = inputs.reshape(1, -1)

        eval_prediction = model.predict(inputs)

        evals_list.append(eval_prediction)

        if board.is_checkmate():
            return move # Always make a move which gives checkmate if possible.

        board.pop()

        # New portion (added 2024-04-09)
        if board.is_capture(move):
            if board.turn:
                evals_list[-1] += 0.25 # Modify to add piece value eventually
            else:
                evals_list[-1] -= 0.25 # Modify to add piece value eventually
    

    evals_list = np.array(evals_list)

    sorted_indices = np.argsort(evals_list)
    
    if board.turn:
        '''
        if it's white's turn, we must reverse the array such that the highest evaluation is first
        if it's black's turn, keep the array ascending such that the lowest evaluation for the white pieces is first
        ''' 
        sorted_indices = sorted_indices[::-1]

    # Use the sorted indices to sort legal_moves and evals_list
    sorted_legal_moves = np.array(legal_moves_list)[sorted_indices]
    sorted_evals_list = evals_list[sorted_indices]

    if not stochastic: # if not using stochastic mode return best move
        return sorted_legal_moves[0]

    sample = np.random.random_sample()

    if sample <= 0.65 or move_number > 7: # 65% chance for best move
        return sorted_legal_moves[0]
    elif sample <= 0.85 or move_number > 5: # 25% chance for second-best move
        return sorted_legal_moves[1]
    elif sample <= 0.975 or move_number > 3: #  7.5% chance for third-best move
        return sorted_legal_moves[2]
    else: # 2.5% chance for fourth-best move
        return sorted_legal_moves[3]
    




def test_game_model(model_loaded):

    board = chess.Board()

    counter = 1
    try:
        while True:
            counter += 1
            time.sleep(1)

            move = predict_SK(model_loaded, board.fen(), move_number=counter)
            
            print(move)
            board.push(move[0])
            print(board)

            print()
    except KeyboardInterrupt:
        print('KeyboardInterrupt: Ended early. Thanks for playing!')




def test_game_models_democracy(models_list):

    board = chess.Board()

    counter = 1
    try:
        while True:
            counter += 1
            time.sleep(1)

            move = predict_move_democracy(models_list, board.fen(), move_number=counter)
            
            print(move)
            board.push(move)
            print(board)

            print()
    except KeyboardInterrupt:
        print('KeyboardInterrupt: Ended early. Thanks for playing!')



def predict_cp_democracy(X_test, l, weights=None):
    """
    Compute the weighted average prediction for a given test set.

    Parameters:
    -----------
    X_test : array-like
        The test set to make predictions on.

    l : list
        A list of loaded models to be evaluated.

    weights : array-like, optional
        An array of weights of the same length as l for a weighted average.
        If not provided, equal weights are assigned to each model.

    Returns:
    --------
    weighted_average : ndarray
        The weighted average prediction.

    Notes:
    ------
    - The models in l should have a `predict` method to make predictions on X_test.
    - If weights are not provided, equal weights are assigned to each model.
    """

    if not weights:
        weights = [1/len(l) for i in range(len(l))]

    predictions = []
    for model in l:
        prediction = model.predict(X_test.reshape(1, -1))
        predictions.append(prediction)

    weighted_average = np.average(predictions, axis=0, weights=weights)

    return weighted_average



def predict_move_democracy(models_list, fen, move_number = 5, stochastic = True, weights = None):
    '''
    ### NOTE TO SELF: models_list should be a list of sklearn regressors with a predict method
    '''

    board = chess.Board(fen)
    legal_moves_list = list(board.legal_moves)
    evals_list = [0] * len(legal_moves_list)
    
    for i, move in enumerate(legal_moves_list):

        board.push(move)

        if board.is_checkmate():
            return move # Always make a move which gives checkmate if possible.

        fen_array = fen_str_to_1d_array(board.fen())

        pieces_counts = get_number_of_pieces(board.fen())

        inputs = np.concatenate((fen_array, pieces_counts))
        inputs = inputs.reshape(1, -1)

        eval_pred_weighted = predict_cp_democracy(inputs, models_list, weights = weights)

        evals_list[i] += np.sum(eval_pred_weighted)

        board.pop()

        # New portion (added 2024-04-09)
        if board.is_capture(move):
            if board.turn:
                evals_list[-1] += 0.25 # Modify to add piece value eventually
            else:
                evals_list[-1] -= 0.25 # Modify to add piece value eventually
    

    evals_list = np.array(evals_list)

    sorted_indices = np.argsort(evals_list)
    
    if board.turn:
        '''
        if it's white's turn, we must reverse the array such that the highest evaluation is first
        if it's black's turn, keep the array ascending such that the lowest evaluation for the white pieces is first
        ''' 
        sorted_indices = sorted_indices[::-1]

    # Use the sorted indices to sort legal_moves and evals_list
    sorted_legal_moves = np.array(legal_moves_list)[sorted_indices]
    sorted_evals_list = evals_list[sorted_indices]

    if not stochastic: # if not using stochastic mode return best move
        return sorted_legal_moves[0]

    sample = np.random.random_sample()

    if sample <= 0.65 or move_number > 7: # 65% chance for best move
        return sorted_legal_moves[0]
    elif sample <= 0.85 or move_number > 5: # 25% chance for second-best move
        return sorted_legal_moves[1]
    elif sample <= 0.975 or move_number > 3: #  7.5% chance for third-best move
        return sorted_legal_moves[2]
    else: # 2.5% chance for fourth-best move
        return sorted_legal_moves[3]

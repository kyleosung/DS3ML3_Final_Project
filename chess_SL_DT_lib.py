import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import chess

import glob

def fen_str_to_1d_array(fen):
    """
    Converts a FEN string representation of a chess board to a 1-d vector array representation.

    Args:
        fen (str): The FEN string representing the chess board.

    Returns:
        np.ndarray: A array vector representation of the chess board.

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

def get_current_version():
    from pathlib import Path
    import pickle

    model_file_path = 'model_RF_1.joblib'
    counter = 1

    while Path(model_file_path).is_file(): # ensure that no files are overwritten
        counter += 1
        model_file_path = f'model_RF_{counter}.joblib'
    
    return counter
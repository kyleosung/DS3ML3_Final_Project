import chess_SL_E8_lib as lib
import chess_SK_lib as SKlib

import numpy as np

import chess
import torch

import time

def predict_ensemble(model, fen, skmodels_list, weights=None, dl_to_sk=0.8, move_number=0, stochastic=True):
    """
    Predicts the evaluation of all legal moves in a given chess position.

    Parameters
    ----------
    model : torch.nn.Module
        trained PyTorch model used for prediction
    fen : str
        FEN string representing the chess position.
    skmodels_list : list
        List of trained scikit-learn models used for prediction
    weights : list
        List of weights for the sk ensemble models
    dl_to_sk : float
        Weight for the deep learning model
    move_number : int, optional
        The number of the move in the game. Default: 0.
    stochastic : bool, optional
        If True, predictions are stochastic (only in the first seven moves)
        If False, predictions are deterministic

    Returns
    -------
    chess.Move object representing the best move to play.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    board = chess.Board(fen)
    legal_moves_list = list(board.legal_moves)
    evals_list = []

    if not weights:
        weights = [1/len(skmodels_list) for i in range(len(skmodels_list))]

    model.eval()
    with torch.no_grad():
        for move in legal_moves_list:

            if board.is_checkmate():
                return move # Always make a move which gives checkmate if possible.
            
            # is_capture = board.is_capture(move)

            board.push(move)
            fen_tensor = lib.fen_str_to_3d_tensor(board.fen()).unsqueeze(0).to(device)
            # print(fen_tensor.shape)

            pieces_counts = lib.get_number_of_pieces(board.fen()).unsqueeze(0).to(device)

            DL_eval = float(model(fen_tensor, pieces_counts).to('cpu'))

            fen_array = SKlib.fen_str_to_1d_array(board.fen())
            pieces_counts_array = SKlib.get_number_of_pieces(board.fen())

            SK_inputs = np.concatenate((fen_array, pieces_counts_array))

            Ensemble_eval = SKlib.predict_cp_democracy(SK_inputs, skmodels_list, weights)

            evals_list.append(DL_eval * dl_to_sk + Ensemble_eval * (1 - dl_to_sk) )



            board.pop()

            # New portion (added 2024-04-09)
            if board.is_capture(move):
                if board.turn:
                    evals_list[-1] += 0.1 # Modify to add piece value eventually
                else:
                    evals_list[-1] -= 0.1 # Modify to add piece value eventually
    

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
    



def __test_ensemble_model(model_DL, skmodels_list, weights=None, dl_to_sk=0.8, move_number=0, stochastic=True):
    board = chess.Board()
    counter = 1
    try:
        while True:
            counter += 1
            time.sleep(1)

            move = predict_ensemble(model_DL, board.fen(), skmodels_list, weights, move_number=counter)
            
            print(move)
            board.push(move)
            print(board)

            print()
    except KeyboardInterrupt:
        print('KeyboardInterrupt: Ended early. Thanks for playing!')
from __future__ import annotations
import chess
from chess.engine import PlayResult, Limit
from chess import Move
import random
from lib.engine_wrapper import MinimalEngine, MOVE
from typing import Any
import logging


# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)


import chess_SL_E4_lib as lib
import torch

model_loaded = torch.load('models/model_E4-1.pth', map_location=torch.device('cpu'))


class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""

    def search(self, board: chess.Board, *args: Any, **xargs: Any) -> PlayResult:

        global model_loaded

        prediction = lib.predict(model_loaded, board.fen())
        
        move = Move.from_uci(prediction)

        return PlayResult(move, None)
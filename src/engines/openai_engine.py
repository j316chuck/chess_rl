import os
import io
import openai
import pandas as pd
import chess
import chess.pgn
import re 
import subprocess

import mcli
import chess
from openai import OpenAI  # Ensure this is the correct import based on your OpenAI client
import openai  # If you're using the standard OpenAI library


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_OPENAI_URL = "https://api.openai.com/v1"

class OpenAIEngine:
    """
    An engine that queries the OpenAI API for the best chess move.
    """
    def __init__(
        self,
        client: OpenAI,
        model_name: str = "model",
        temperature: float = 0.0,
    ):
        """
        Initialize the OpenAIEngine with a custom OpenAI client.

        :param client: An instance of the OpenAI client.
        :param model_name: The name of the model to use.
        :param temperature: Sampling temperature.
        """
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
    
    def _fallback_move(self, board: chess.Board) -> chess.Move:
        """
        Return the first legal move as a fallback.

        :param board: The current state of the chess board.
        :return: A fallback chess move in UCI notation.
        :raises ValueError: If there are no legal moves available.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available.")
        fallback_move = legal_moves[0]
        print(f"Falling back to first legal move: {fallback_move.uci()}")
        return fallback_move
    
    def play(self, board: chess.Board) -> chess.Move:
        """
        Determine and return the best move in UCI notation using the OpenAI API.

        :param board: The current state of the chess board.
        :return: The best chess move in UCI notation.
        :raises ValueError: If the API response is invalid or the move is illegal.
        """
        # Convert the board to FEN or board_fen based on configuration
        board_state = board.fen()
        
        # Construct the prompt
        prompt_content = (
            f"Find the best UCI chess move for the following FEN position: {board_state}"
        )
        
        messages = [
            {
                "role": "user",
                "content": prompt_content
            }
        ]
        
        # Make the API call using the provided client
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=messages,
            )
        except Exception as e:
            print(f"Failed to connect to OpenAI API: {e}")
            return self._fallback_move(board)
        
        # Extract and clean the response
        raw_reply = response.choices[0].message.content.strip()
        
        # Attempt to parse the move
        try:
            move = chess.Move.from_uci(raw_reply)
            if move in board.legal_moves:
                return move
            else:
                print(f"Received move '{move}' is not legal.")
                return self._fallback_move(board)
        except ValueError:
            print(f"Could not parse UCI move from response: '{raw_reply}'")
            return self._fallback_move(board)

def evaluate_puzzle_from_pandas_row(puzzle: pd.Series, engine: OpenAIEngine) -> bool:
    """
    Check if the engine solves the puzzle. 
    Lichess puzzle logic:
      1) puzzle['PGN'] is the position's moves from the puzzle start to the end.
      2) puzzle['Moves'] is the solution sequence of moves in UCI format separated by spaces.
    """
    game = chess.pgn.read_game(io.StringIO(puzzle['PGN']))
    if game is None:
        raise ValueError(f'Failed to read game from PGN {puzzle["PGN"]}.')
    board = game.end().board()
    solution_moves = puzzle['Moves'].split(' ')
    print("Fen string: ", board.fen())
    print("Solution moves: ", solution_moves)
    # Reproduce the puzzle logic: 
    # Lichess puzzle is from the perspective after applying the first move.
    # Then we check if our move matches the solution for each user move.
    for move_idx, move_uci in enumerate(solution_moves):
        # If it's our turn (move_idx % 2 == 1), we query the engine
        if move_idx % 2 == 1:
            predicted_uci = engine.play(board).uci()
            if predicted_uci != move_uci:
                # If the predicted move is a mate-in-1 anyway, that is acceptable
                board.push(chess.Move.from_uci(predicted_uci))
                return board.is_checkmate()
        # Always push the correct solution move
        board.push(chess.Move.from_uci(move_uci))
    return True

def main(model_name: str = "gpt-3.5-turbo", api_key: str=BASE_OPENAI_API_KEY, base_url: str = BASE_OPENAI_URL, n_puzzles: int = 100, update_metadata: bool = False):
    # Suppose we have a puzzles CSV with columns: 'PGN', 'Moves', 'Rating', ...
    puzzles_path = os.path.join(CURRENT_DIR, "./puzzles.csv")
    puzzles = pd.read_csv(puzzles_path, nrows=n_puzzles)  # e.g., read 10 puzzles

    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    # Create our engine
    engine = OpenAIEngine(
        client=client,
        model_name=model_name,  # or "gpt-4", etc.
    )

    # Evaluate puzzles in parallel using ThreadPoolExecutor
    results = []
    for puzzle_id, puzzle in puzzles.iterrows():
        try:
            print(f"Evaluating puzzle: {puzzle_id}")
            correct = evaluate_puzzle_from_pandas_row(puzzle, engine)
            results.append((puzzle_id, correct, puzzles.loc[puzzle_id, 'Rating']))
        except Exception as e:
            print(f"Puzzle {puzzle_id} raised an exception: {e}")
            results.append((puzzle_id, False, puzzles.loc[puzzle_id, 'Rating']))

    # Print the results
    for puzzle_id, correct, rating in results:
        print({
            'puzzle_id': puzzle_id,
            'correct': correct,
            'rating': rating,
        })

    # Print accuracy as a percentage
    accuracy = sum(correct for _, correct, _ in results) / len(results) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    if update_metadata:
        metadata = {
            "mosaicml/eval/puzzle_accuracy": accuracy,
            "mosaicml/eval/n_puzzles": n_puzzles,
        }
        run_name = os.environ.get("RUN_NAME", "default_run_name")
        mcli.update_run_metadata(run_name, metadata)

def download_s3_path_awscli(
    s3_uri: str,
    out_path: str,
    max_concurrent_requests: int = 200,
    quiet: bool = False,
    **kwargs,
):
    if not re.match("s3:.*", s3_uri):
        raise ValueError

    s3_uri = s3_uri.rstrip("/")
    flags = []
    if "AWS_PROFILE" in os.environ:
        flags += ["--profile", os.environ["AWS_PROFILE"]]

    kwargs = {}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL

    subprocess.run(
        [
            "aws",
            "configure",
            "set",
            "s3.max_concurrent_requests",
            str(max_concurrent_requests),
            *flags,
        ],
        check=True,
    )
    subprocess.run(["aws", "s3", "sync", s3_uri, out_path, *flags], **kwargs, check=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run chess puzzle evaluation with OpenAI API')
    parser.add_argument('--model-name', type=str, default='gpt-3.5-turbo',
                        help='OpenAI model name to use')
    parser.add_argument('--api-key', type=str, default=BASE_OPENAI_API_KEY,
                        help='OpenAI API key')
    parser.add_argument('--base-url', type=str, default=BASE_OPENAI_URL,
                        help='OpenAI API base URL')
    parser.add_argument('--n-puzzles', type=int, default=100,
                        help='Number of puzzles to evaluate')
    parser.add_argument('--update-metadata', action='store_true',
                        help='Whether to update run metadata')
    parser.add_argument('--puzzles-path', type=str, default=os.path.join(CURRENT_DIR, "./puzzles.csv"),
                        help='Path to the puzzles CSV file')

    args = parser.parse_args()
    # add s3 download option for puzzles.csv if puzzles_path is not a local path
    if not os.path.exists(args.puzzles_path) and re.match("s3:.*", args.puzzles_path):
        download_s3_path_awscli(s3_uri=args.puzzles_path, out_path=os.path.join(CURRENT_DIR, "./puzzles.csv"))
    print(args.puzzles_path)

    main(
        model_name=args.model_name,
        api_key=args.api_key, 
        base_url=args.base_url,
        n_puzzles=args.n_puzzles,
        update_metadata=args.update_metadata
    )

import os
import io
import openai
import pandas as pd
import chess
import chess.pgn
import chess.svg
import re 
import subprocess
import mcli

from chess import Board
from cairosvg import svg2png


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PUZZLES_PATH = os.path.join(CURRENT_DIR, "./puzzles.csv")
BASE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_OPENAI_URL = "https://api.openai.com/v1"

def visualize_chess_image(fen: str, line1: str=None, line2: str=None, output_path: str=None, puzzle_id=None, correct: bool=False):
    """
    Generate a chess image with the given FEN, move lines, and title.

    Args:
    - fen (str): FEN string of the chess position.
    - line1 (str): Solution line to display (green).
    - line2 (str): Model line to display (blue).
    - output_path (str): File path to save the generated PNG image.
    - puzzle_id (str): Optional puzzle ID to include in the title.
    - correct (bool): if puzzle solution is correct. 
    """
    # Initialize the chessboard from the FEN
    board = Board(fen)

    # Create an SVG representation of the board
    board_svg = chess.svg.board(board, size=600)

    # Generate title if puzzle_id is provided
    title_svg = ""
    if puzzle_id:
        title_svg = f'<text x="10" y="25" font-family="Arial" font-size="18" fill="black">Puzzle ID: {puzzle_id}</text>'

    # Combine board and lines into SVG
    correct_color = "green" if correct else "red"
    correct_msg = "Correct" if correct else "Incorrect"
    correct_svg = f'<text x="10" y="725" font-family="Arial" font-size="16" fill="{correct_color}">Result: {correct_msg}</text>'

    lines_svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="600" height="740">
      <rect width="100%" height="100%" fill="white" />
      <g transform="translate(0,40)">
        {board_svg}
      </g>
      {title_svg}
      <text x="10" y="665" font-family="Arial" font-size="16" fill="green">Puzzle Solution: {line1}</text>
      <text x="10" y="695" font-family="Arial" font-size="16" fill="blue">Model Prediction: {line2}</text>
      {correct_svg}
    </svg>
    """

    # Convert the SVG to a PNG file
    if output_path:
        svg2png(bytestring=lines_svg, write_to=output_path)
        print(f"Image saved to {output_path}")

class OpenAIEngine:
    """
    An engine that queries the OpenAI API for the best chess move.
    """
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str = "model",
        temperature: float = 0.0,
    ):
        """
        Initialize the OpenAIEngine with a custom OpenAI client.

        :param api_key: The API key to use.
        :param base_url: The base URL to use.
        :param model_name: The name of the model to use.
        :param temperature: Sampling temperature.
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
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
        # Convert the board to FEN
        board_state = board.fen()
        
        # Construct the prompt
        prompt_content = (
            f"You are an expert chess player. Find the best UCI chess move for the following FEN position: {board_state}"
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
                n=1,
                messages=messages,
                temperature=self.temperature,
            )
        except Exception as e:
            print(f"Failed to connect to OpenAI API: {e}")
            return self._fallback_move(board)
        
        # Extract and clean the response
        raw_reply = response.choices[0].message.content.strip()
        print("Predicted raw reply: ", raw_reply)
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

def evaluate_puzzle_from_pandas_row(puzzle: pd.Series, puzzle_idx: str, engine: OpenAIEngine, save_folder: str=None) -> bool:
    """
    Check if the engine solves the puzzle. 
    Lichess puzzle logic:
      1) puzzle['PGN'] is the position's moves from the puzzle start to the end.
      2) puzzle['Moves'] is the solution sequence of moves in UCI format separated by spaces.
    """
    print(puzzle)
    game = chess.pgn.read_game(io.StringIO(puzzle['PGN']))
    if game is None:
        raise ValueError(f'Failed to read game from PGN {puzzle["PGN"]}.')
    board = game.end().board()
    solution_moves = puzzle['Moves'].split(' ')
    puzzle_id = f"{puzzle['PuzzleId']}" # get the index and puzzle id 

    solution_board = board.copy()
    # Reset to original board for model moves
    model_moves = []
    is_correct = True
    for move_idx, move_uci in enumerate(solution_moves):
        turn = "opponent" if move_idx % 2 == 0 else "player"
        print(f"Expected {turn} move: {move_uci}")
        
        # If it's our turn, query the engine
        if move_idx % 2 == 1:
            predicted_uci = engine.play(board).uci()
            move_is_correct = predicted_uci == move_uci
            print(f"Predicted player move `{predicted_uci}` is correct? {move_is_correct}")
            if not move_is_correct:
                model_moves.append(predicted_uci)
                is_correct = False
                break
        # Always push the correct solution move to advance the board state
        board.push(chess.Move.from_uci(move_uci))
        model_moves.append(move_uci)
    
    if save_folder:
        line1 = " ".join(str(x) for x in solution_moves)
        line2 = " ".join(str(x) for x in model_moves)
        output_path = os.path.join(save_folder, f"{puzzle_idx}.png")
        os.makedirs(save_folder, exist_ok=True)
        print("Visualizing chess puzzle", puzzle_idx)
        visualize_chess_image(fen=solution_board.fen(), line1=line1, line2=line2, output_path=output_path, puzzle_id=puzzle_id, correct=is_correct)
    return is_correct

def main(model_name: str = "gpt-3.5-turbo", api_key: str=BASE_OPENAI_API_KEY, base_url: str = BASE_OPENAI_URL, n_puzzles: int = 100, update_metadata: bool = False, temperature: float = 0.0, save_folder: str = None, remote_save_folder: str = None):
    # Suppose we have a puzzles CSV with columns: 'PGN', 'Moves', 'Rating', ...
    puzzles = pd.read_csv(PUZZLES_PATH, nrows=n_puzzles)  # e.g., read 10 puzzles

    # Create our engine
    engine = OpenAIEngine(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,  # or "gpt-4", etc.
        temperature=temperature,
    )

    # Evaluate puzzles iteratively
    results = []
    for puzzle_idx, puzzle in puzzles.iterrows():
        try:
            print("--------------------------------")
            print(f"Evaluating puzzle: {puzzle_idx}")
            correct = evaluate_puzzle_from_pandas_row(puzzle, puzzle_idx, engine, save_folder)
            results.append((puzzle_idx, correct, puzzles.loc[puzzle_idx, 'Rating']))
        except Exception as e:
            print(f"Puzzle {puzzle_idx} raised an exception: {e}")
            results.append((puzzle_idx, False, puzzles.loc[puzzle_idx, 'Rating']))

    if save_folder and remote_save_folder: 
        upload_s3_path_awscli(save_folder, remote_save_folder)

    # Print the results
    for puzzle_idx, correct, rating in results:
        print({
            'puzzle_idx': puzzle_idx,
            'correct': correct,
            'rating': rating,
        })

    # Print accuracy as a percentage
    accuracy = sum(correct for _, correct, _ in results) / len(results) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    # Update mcli run metadata
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
    print("Downloading s3 path", s3_uri, "to", out_path)
    subprocess.run(["aws", "s3", "cp", s3_uri, out_path, *flags], **kwargs, check=True)
    assert os.path.exists(out_path)
    
def upload_s3_path_awscli(
    upload_path: str,
    s3_uri: str,
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
    recursive = "--recursive" if os.path.isdir(upload_path) else ""
    subprocess.run(["aws", "s3", "cp", recursive, upload_path, s3_uri, *flags], **kwargs, check=True)

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
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature')
    parser.add_argument('--save_folder', type=str, default=os.path.join(CURRENT_DIR, "./results"),
                        help='Path to the folder to save results')
    parser.add_argument('--remote_save_folder', type=str, default=None,
                        help='Path to the remote folder to upload results to')
    
    args = parser.parse_args()
    # download s3 path to local puzzles.csv if it's not already there
    if not os.path.exists(PUZZLES_PATH):
        download_s3_path_awscli(s3_uri=args.puzzles_path, out_path=PUZZLES_PATH)
    
    main(
        model_name=args.model_name,
        api_key=args.api_key, 
        base_url=args.base_url,
        n_puzzles=args.n_puzzles,
        update_metadata=args.update_metadata,
        temperature=args.temperature,
        save_folder=args.save_folder,
        remote_save_folder=args.remote_save_folder,
    )

from chess import Board
import chess.svg
from cairosvg import svg2png

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

if __name__ == "__main__":
    # Example usage
    fen_example = "r1bqkbnr/pppppppp/n7/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1"
    line1_example = "e4 d6 d4 Nf6 Nc3 g6"
    line2_example = "e4 Nc6 Bb5 d6 O-O"
    output_path_example = "chess_position_output_final.png"

    # Generate the chess image
    generate_chess_image(fen_example, line1_example, line2_example, output_path_example, puzzle_id="12345")

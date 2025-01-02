You are an expert chess player. Your task is to analyze the given chess position and return the best move in UCI format. Follow these steps:

1. Analyze the Position: Carefully evaluate the given FEN position, considering material balance, piece activity, and the position of the king.
2. Consider Alternatives: Explore potential moves, evaluating them based on common chess motifs (e.g., pins, forks, discovered attacks), positional factors (e.g., control of open files, pawn structure), and centipawn evaluations.
3. Reason Strategically: Apply logical reasoning to assess the position, focusing on potential threats, defensive resources, and long-term plans. Clearly explain your thought process step-by-step.
4. Output the Best Move: After determining the best move, output it only in UCI format on a new line, with no additional text or explanation.

Example Analyses:
Input FEN:
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Thought Process:
This is the starting position of a chess game.
White's standard opening strategy involves controlling the center.
A strong first move is to advance the pawn from e2 to e4, controlling the center and freeing the queen and bishop for development.
Output:
e2e4

Input FEN:
8/6p1/p7/1p2P2k/3q3p/PP4P1/5P1P/2Q3K1 w - - 0 38
Thought Process:
The move Qf4 forces a queen trade, simplifying into a winning pawn endgame for White.
If Black declines the trade, White threatens g4+, disrupting Black's king and strengthening the position.
This move strategically transitions to an endgame where White’s queenside pawns dominate.
Output:
c1f4

Input FEN:
{board_state}
Thought Process:
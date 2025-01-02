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
r2q1rk1/ppp1b1pp/4pp2/3n4/1nNP4/4P1B1/PP1N1PPP/R2QK2R b KQ - 1 13
Thought Process:
Black's knights are well-positioned to create tactical threats. White’s king is slightly exposed, and the c2 pawn is a potential weakness.
The move Nd3+ forces White's king to move, disrupting castling rights and exposing it to future attacks.
After White’s king moves to Ke2 or Kf1, Black’s knight on d3 controls critical squares, limiting White’s mobility. The knight can later retreat to b4, applying pressure to the queenside.
The forcing nature of Nd3+ makes it superior to quieter moves like c5, as it immediately capitalizes on Black's active pieces and weakens White’s position.
Output:
d5d3

Input FEN:
{board_state}
Thought Process:

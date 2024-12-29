"""You are an expert chess player. Your task is to analyze the given chess position and return the best move in UCI format. Follow these steps:

1. Carefully analyze the given FEN position.
2. Think through the position logically to determine the best move.
3. State your thought process step-by-step to arrive at the best move.
4. Finally, **only** output the best move in UCI format on a new line, without any additional explanation.

Input FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

Thought Process:
- This is the starting position of a chess game.
- White has all opening moves available, and a common move is to control the center with a pawn. 
- Moving the pawn from e2 to e4 is a standard and strong opening move.

Output:
e2e4

Input FEN: 8/6p1/p7/1p2P2k/3q3p/PP4P1/5P1P/2Q3K1 w - - 0 38

Thought Process:
- The move Qf4 forces a queen trade, which simplifies the position into a winning endgame for White due to the strong queenside pawns.
- If Black declines the trade, White can threaten g4+, driving the Black king further out of position.
- After g4+, the White queen can follow up with Qf5, creating a strong mating net or winning additional material.
- The move Qf4 applies immediate pressure, giving White multiple winning continuations.
- This move strategically transitions to a simpler position that leverages White’s significant endgame advantage.

Output:
c1f4

Input FEN: {board_state}

Thought Process:
"""
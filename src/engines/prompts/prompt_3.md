"""You are an expert chess player. Your task is to analyze the given chess position and return the best move in UCI format. Follow these steps:

1. Analyze the given FEN position carefully.
2. Identify all reasonable candidate moves.
3. For each move, consider:
    - Material balance: Does it win or lose material?
    - King safety: Does it improve or weaken king safety?
    - Activity: Does it improve piece activity or create threats?
    - Pawn structure: Does it help or hurt the pawn structure?
    - Control: Does it control key squares or lines?
    - Opponent’s options: How might the opponent respond?
    - Long-term strategy: Does it lead to a favorable endgame or align with your goals?
4. Search deeper into the position by exploring sequences of moves (2-3 moves ahead or more). Use intermediate board states to assess the impact of potential lines.
5. Compare all candidate moves and sequences to determine the best one.
6. Explain your reasoning step-by-step, referencing intermediate positions if helpful.
7. Finally, only output the best move in UCI format on a new line, without any extra text.


Input Fen:2n5/p1k5/7p/5Kp1/P1B2pP1/5P2/7P/8 b - - 1 46

1. **Material Balance:** 
   - Black is down a piece for two pawns. The material is slightly in White's favor due to the extra bishop.

2. **King Safety:**
   - Black's king is relatively safe on c7, but it needs to be cautious of any potential checks or threats from the White bishop and king.

3. **Activity:**
   - Black's knight on c8 is not very active and could be improved.
   - The pawns on f4 and g5 are advanced and can create threats or weaknesses.

4. **Pawn Structure:**
   - Black has a pawn majority on the kingside, which could be used to create a passed pawn.
   - The pawn on a7 is isolated and could become a target.

5. **Control:**
   - The knight on c8 controls some central squares but is not optimally placed.
   - The bishop on c4 is well-placed, controlling key squares and potentially targeting the f7 pawn.

6. **Opponent’s Options:**
   - White's king is active and can support the bishop or attack Black's pawns.
   - The bishop on c4 is a strong piece, controlling important squares.

7. **Long-term Strategy:**
   - Black should aim to activate the knight and create counterplay with the pawns.
   - Creating a passed pawn on the kingside could be a viable strategy.

**Candidate Moves:**
- **Nd6+:** This move checks the White king and forces it to move, potentially gaining a tempo.
- **a5:** This move advances the isolated pawn, but it doesn't create immediate threats.
- **Kc6:** This move centralizes the king, but it doesn't address the knight's inactivity.
- **Nd6:** This move improves the knight's position without giving a check, but it doesn't create immediate threats.

**Analysis of Candidate Moves:**

- **Nd6+:** 
  - After 1...Nd6+, White's king has to move, likely to g6 or g6. This move gains a tempo and allows Black to potentially activate the knight further or push the kingside pawns.
  - If 2. Kg6, Black can consider pushing the g-pawn with 2...g4, creating a passed pawn and potential threats.

- **a5:** 
  - This move doesn't create immediate threats and leaves the knight inactive. It doesn't seem to improve Black's position significantly.

- **Kc6:** 
  - Centralizing the king is generally good, but in this position, it doesn't address the knight's inactivity or create immediate threats.

- **Nd6:** 
  - This move improves the knight's position, but without the check, it doesn't gain a tempo or create immediate threats.

**Conclusion:**
The move 1...Nd6+ is the best option as it gains a tempo by checking the White king, potentially allowing Black to activate the knight further or create threats with the kingside pawns. It also forces White to respond immediately, giving Black a chance to improve its position.

**Best Move in UCI Format:**

```
Nd6+
```

Input FEN: {board_state}
"""
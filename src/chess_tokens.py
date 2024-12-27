from itertools import product

def get_all_uci_chess_moves():
    # Define all squares on the board
    files = "abcdefgh"
    ranks = "12345678"
    squares = [f + r for f, r in product(files, ranks)]

    # Generate all combinations of moves (source -> destination)
    basic_moves = [f"{src}{dest}" for src, dest in product(squares, repeat=2)]

    # Add promotion moves
    promotion_suffixes = ['q', 'r', 'b', 'n']
    promotion_moves = []
    for file in files:
        # White pawn promotions
        promotion_moves += [f"{file}7{file}8{suffix}" for suffix in promotion_suffixes]
        # Black pawn promotions
        promotion_moves += [f"{file}2{file}1{suffix}" for suffix in promotion_suffixes]

    # Add castling moves
    castling_moves = ["e1g1", "e1c1", "e8g8", "e8c8"]

    # Combine all move types
    all_moves = basic_moves + promotion_moves + castling_moves
    return all_moves

if __name__ == "__main__":
    all_uci_chess_moves = get_all_uci_chess_moves()
    print(f"Total possible UCI chess moves: {len(all_uci_chess_moves)}")
    serialize = False
    if serialize:
        import pickle 
        pickle.dump(all_uci_chess_moves, open('all_uci_chess_moves_4164.pkl', 'wb'))
    print("Number of moves", len(all_uci_chess_moves))
    assert len(all_uci_chess_moves) == 4164

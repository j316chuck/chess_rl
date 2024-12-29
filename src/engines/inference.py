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
import os 
import json 

from datasets import Dataset
from chess import Board
from cairosvg import svg2png

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_OPENAI_URL = "https://api.openai.com/v1"


class LLMChessMove:
    def __init__(self, move: chess.Move, is_legal: bool, raw_reply: str, is_correct: bool):
        self.move = move
        self.is_legal = is_legal
        self.raw_reply = raw_reply
        self.is_correct = is_correct

class LLMChessEngine:
    """
    An engine that queries the OpenAI API for the best chess move.
    """
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str = "model",
        temperature: float = 0.0,
        prompt: list[str] = None,
        n_samples: int = 1,
    ):
        """
        Initialize the OpenAIEngine with a custom OpenAI client.

        :param api_key: The API key to use.
        :param base_url: The base URL to use.
        :param model_name: The name of the model to use.
        :param temperature: Sampling temperature.
        :param prompt: Prompt to use for the engine.
        :param n_samples: Number of samples to sample from the model.
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model_name = model_name
        self.temperature = temperature
        self.prompt = prompt
        self.n_samples = n_samples
        
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

    def extract_uci_move(self, raw_reply: str) -> str:
        """
        Extract the UCI move from the raw reply by getting the last 4 - 5 characters.
        """
        return raw_reply[-5:].strip()
    
    def play(self, board: str, expected_move: str) -> LLMChessMove:
        """
        Determine and return the best move in UCI notation using the OpenAI API.

        :param board: The current state of the chess board in FEN
        :return: The best chess move in UCI notation.
        :raises ValueError: If the API response is invalid or the move is illegal.
        """        
        board_state = board.fen()

        # Construct the prompt
        if self.prompt:
            prompt_content = self.prompt.format(board_state=board_state)
        else:
            prompt_content = (
                f"Find the best UCI chess move for the following FEN position: {board_state}"
            )
        messages = [
            {
                "role": "user",
                "content": prompt_content
            }
        ]   

        llm_move = LLMChessMove(
            raw_reply=None,
            move=None,
            is_legal=False,
            is_correct=False,
        )
        # Make the API call using the provided client
        try:
            raw_replies = []
            for _ in range(self.n_samples):
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    n=1,
                    messages=messages,
                    temperature=self.temperature,
                )
                raw_replies.append(response.choices[0].message.content.strip())
            response = self.client.chat.completions.create(
                model=self.model_name,
                n=self.n_samples,
                messages=messages,
                temperature=self.temperature,
            )
        except Exception as e:
            print(f"Failed to connect to OpenAI API: {e}")
            return llm_move
        
        # Extract and clean the response
        llm_moves = []
        for raw_reply in raw_replies:
            llm_move.raw_reply = raw_reply

            # print("Predicted raw reply: ", llm_move.raw_reply)
            llm_move.move = self.extract_uci_move(llm_move.raw_reply)
            # Attempt to parse the move
            try:
                move = chess.Move.from_uci(llm_move.move)
                if move in board.legal_moves:
                    llm_move.is_legal = True
                    llm_move.move = move.uci()
                    llm_move.is_correct = str(llm_move.move) == expected_move
                    llm_moves.append(llm_move)
                else:
                    # print(f"Received move '{move}' is not legal.")
                    llm_moves.append(llm_move)
            except ValueError:
                # print("Could not parse UCI move from response")
                llm_moves.append(llm_move)
        return llm_moves


class ChessInferenceEngine:
    def __init__(self, model_name: str, api_key: str, base_url: str, temperature: float, prompt: str, n_samples_per_position: int):
        """
        Initialize the chess inference engine with the given parameters.

        Args:
            model_name (str): The name of the model to use for inference.
            api_key (str): The API key to use for inference.
            base_url (str): The base URL to use for inference.
            temperature (float): Temperature to sample from the model.
            prompt (str): The prompt to use for inference.
            n_samples_per_position (int): Number of samples to sample from the model for each position.
        """
        self.engine = LLMChessEngine(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,  # or "gpt-4", etc.
            temperature=temperature,
            prompt=prompt,
            n_samples=n_samples_per_position,
        )

    def sample_moves(self, fen_positions: list[str], golden_moves: list[str], n_data_points_per_position: int, dpo: bool, save_folder: str, remote_save_folder: str, max_samples: int=None) -> list[str]:
        """
        Sample moves from the model for each of the n_points.
        """
        correct_moves = []
        incorrect_moves = []
        prompt_responses = []
        answer_responses = []
        os.makedirs(save_folder, exist_ok=True)

        for ii, (fen_position, correct_move) in enumerate(zip(fen_positions, golden_moves)):
            print("--------------------------------")
            print(f"Sampling move {ii} of {len(fen_positions)}.")
            board = chess.Board(fen_position)
            llm_moves = self.engine.play(board, correct_move)
            incorrect_move_responses = [llm_move.raw_reply for llm_move in llm_moves if not llm_move.is_correct]
            correct_move_responses = [llm_move.raw_reply for llm_move in llm_moves if llm_move.is_correct]
            percentage_correct = len(correct_move_responses) / (len(incorrect_move_responses) + len(correct_move_responses)) * 100

            print(f"Percentage correct: {percentage_correct:.2f}%")
            with open(os.path.join(save_folder, f"position_{ii}_percentage_correct_{percentage_correct:.2f}.txt"), "w") as f:
                f.write(f"Percentage correct: {percentage_correct:.2f}%")
            with open(os.path.join(save_folder, f"position_{ii}_incorrect.txt"), "w") as f:
                for i, move in enumerate(incorrect_move_responses):
                    f.write(f"Raw reply {i}:\n{move}\n")
            with open(os.path.join(save_folder, f"position_{ii}_correct.txt"), "w") as f:
                for i, move in enumerate(correct_move_responses):
                    f.write(f"Raw reply {i}:\n{move}\n")
            if dpo:
                import itertools
                import random
                pairs = list(itertools.product(incorrect_move_responses, correct_move_responses))
                num_data_points = min(n_data_points_per_position, len(pairs))
                data_points = random.sample(pairs, k=num_data_points)
                incorrect_moves.extend([pair[0] for pair in data_points[:num_data_points]])
                correct_moves.extend([pair[1] for pair in data_points[:num_data_points]])
                prompt_responses.extend([fen_position for _ in range(num_data_points)])
                answer_responses.extend([correct_move for _ in range(num_data_points)])
            else:
                num_data_points = min(n_data_points_per_position, len(correct_move_responses))
                correct_moves.extend(correct_move_responses[:num_data_points])
                prompt_responses.extend([fen_position for _ in range(num_data_points)])
                answer_responses.extend([correct_move for _ in range(num_data_points)])
                incorrect_moves.extend([incorrect_move_responses[i] for i in range(n_data_points_per_position - num_data_points)])
                # print("Length of prompt responses: ", len(prompt_responses))
                # print("Length of answer responses: ", len(answer_responses))
            print(f"Seen {len(correct_moves)} correct moves. Seen {len(incorrect_moves)} incorrect moves.")
            print(f"{len(correct_moves) / (len(incorrect_moves) + len(correct_moves)) * 100:.2f}% correct moves.")
            if max_samples and ii >= max_samples:
                break

        assert len(prompt_responses) == len(answer_responses), f"Length of prompt responses: {len(prompt_responses)} is not equal to length of answer responses: {len(answer_responses)}"
        assert len(correct_moves) == len(prompt_responses), f"Length of correct move responses: {len(correct_moves)} is not equal to length of prompt responses: {len(prompt_responses)}"
        if dpo:
            assert len(incorrect_moves) == len(correct_moves), f"Length of incorrect move responses: {len(incorrect_moves)} is not equal to length of correct move responses: {len(correct_moves)}"
        print("Total number of data points sampled: ", len(answer_responses))
        # save the data to the save_folder
        if dpo:
            ds = Dataset.from_dict({
                'prompt': prompt_responses,
                'answer': answer_responses,
                'chosen': correct_moves,
                'rejected': incorrect_moves,
            }).train_test_split(test_size=0.1)
            ds['train'].to_json(os.path.join(save_folder, "train.jsonl"))
            ds['test'].to_json(os.path.join(save_folder, "val.jsonl"))
        else:
            ds = Dataset.from_dict({
                'prompt': prompt_responses,
                'response': correct_moves,
                'answer': answer_responses,
            }).train_test_split(test_size=0.1)
            ds['train'].to_json(os.path.join(save_folder, "train.jsonl"))
            ds['test'].to_json(os.path.join(save_folder, "val.jsonl"))
        if remote_save_folder:
            upload_s3_path_awscli(s3_uri=remote_save_folder, out_path=save_folder)
            print("Uploaded data to ", remote_save_folder)
        return ds

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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run chess puzzle evaluation with OpenAI API')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--base_url', type=str, required=True)
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--prompt', type=str, required=False, default=None)
    parser.add_argument('--n_samples_per_position', type=int, required=True)
    parser.add_argument('--base_dataset_path', type=str, required=True)
    parser.add_argument('--n_data_points_per_position', type=int, required=True)
    parser.add_argument('--dpo', action='store_true', help='Enable DPO mode')
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--remote_save_folder', type=str, required=False)
    parser.add_argument('--max_positions', type=int, required=False)
    parser.add_argument('--prompt_path', type=str, required=False, default=None)
    args = parser.parse_args()
    print(args)

    os.makedirs(args.save_folder, exist_ok=True)
    if not os.path.exists(args.base_dataset_path):
        if args.base_dataset_path.startswith("s3://"):
            dataset_path = os.path.join(args.save_folder, "base_dataset.jsonl")
            download_s3_path_awscli(s3_uri=args.base_dataset_path, out_path=dataset_path)
        else:
            raise ValueError(f"Base dataset path {args.base_dataset_path} is not a valid s3 path")
    else: 
        dataset_path = args.base_dataset_path
    dataset = Dataset.from_json(dataset_path)
    print(dataset)
    if not args.prompt:
        assert args.prompt_path, "Prompt path is required if prompt is not provided"
        with open(args.prompt_path, "r") as f:
            args.prompt = f.read()

    engine = ChessInferenceEngine(
        model_name=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        prompt=args.prompt,
        n_samples_per_position=args.n_samples_per_position,
    )
    engine.sample_moves(
        fen_positions=dataset["prompt"],
        golden_moves=dataset["response"],
        n_data_points_per_position=args.n_data_points_per_position,
        dpo=args.dpo,
        save_folder=args.save_folder,
        remote_save_folder=args.remote_save_folder,
        max_samples=args.max_positions, 
    )

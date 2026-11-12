"""``ResponseSchema`` — the columns of a responses parquet: one generated response and its reward per prompt."""


class ResponseSchema:
    """Column names and file names of the responses files an evaluation pass writes."""

    # Key of the prompt, as in the prompts files.
    KEY = "prompt_id"
    # The policy's response, plain text.
    RESPONSE = "response"
    # The frozen reward model's score of the prompt and the response.
    REWARD = "reward"
    # Generated tokens of the response, the end token included.
    RESPONSE_TOKENS = "response_tokens"
    # The split the prompt belongs to.
    SPLIT = "split"

    COLUMNS = (KEY, RESPONSE, REWARD, RESPONSE_TOKENS, SPLIT)

    @staticmethod
    def file_name(split: str) -> str:
        """The parquet file name of one split: ``responses_<split>.parquet``."""
        return f"responses_{split}.parquet"

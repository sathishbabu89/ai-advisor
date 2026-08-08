from dataclasses import dataclass


@dataclass
class Prompt:
    """
    Represents a prompt sent to an LLM.

    Attributes
    ----------
    system_prompt:
        Defines the model behaviour.

    user_prompt:
        Contains the retrieved context and user question.
    """

    system_prompt: str

    user_prompt: str
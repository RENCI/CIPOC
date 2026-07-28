from enum import Enum

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"

    @classmethod
    def _to_int(cls, obj: "ConfidenceLevel") -> int:
        return {level: i for i, level in enumerate(cls)}[obj]
    
    def __int__(self) -> int:
        return self._to_int(self)

    def __lt__(self, other: "ConfidenceLevel"):
        return int(self) < int(other)
    
    def __gt__(self, other: "ConfidenceLevel"):
        return int(self) > int(other)

    @property
    def definition(self) -> str:
        return {
            ConfidenceLevel.LOW: "The answer is a best guess with significant uncertainty.",
            ConfidenceLevel.MEDIUM: "The answer is reasonably supported by the context, but relies on inferences to fill information gaps.",
            ConfidenceLevel.HIGH: "The answer is not explicitly stated, but is strongly supported with clear evidence from the context.",
            ConfidenceLevel.MAX: "The answer is explicitly stated in the provided context.",
        }[self]

def confidence_instructions() -> str:
    lines = [f"- {level.value}: {level.definition}" for level in ConfidenceLevel]
    return "Rate your confidence using one of the following levels:\n" + "\n".join(lines)


def confidence_field(**kwargs):
    """Pydantic field for a confidence value, with the level definitions in its description."""
    return Field(
        description="Level of confidence in the reported answer.\n" + confidence_instructions(),
        **kwargs,
    )

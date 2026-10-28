from pathlib import Path

import httpx
import yaml
import pandas as pd
from transforms.api import LightweightInput, LightweightOutput, transform, Input, Output
from palantir_models.transforms import OpenAiGptChatLanguageModelInput
from palantir_models.models import OpenAiGptChatLanguageModel

from cipoc.agents import OrchestratorAgent
from cipoc.llm import FoundryModelsTransport
from cipoc.models import OrchestratorRunError
from cipoc.utils import CipocConfig

from myproject import resources
from myproject.foundry_models import to_foundry_request, to_openai_response

RESOURCES = Path(resources.__file__).parent
NOTES_DATASET_RID = "<input rid>"  # one row per note
OUTPUT_DATASET_RID = "<output rid>"
MODEL_RID = "<model rid>"

STRING_COLUMNS = {"status": "string", "error": "string", "result_json": "string"}


def build_config(client: httpx.Client) -> CipocConfig:
    raw = yaml.safe_load((RESOURCES / "config.yaml").read_text())
    raw["documents"] = {
        "documents_path": str(RESOURCES),
        "data_dictionary_path": str(RESOURCES / "naaccr_data_dictionary_v25.json"),
        "site_data_dictionary_path": str(RESOURCES / "cipoc_data_dictionary.json"),
        "variable_groups_path": str(RESOURCES / "variable_groups.json"),
    }
    return CipocConfig(raw, model_kwargs={"http_client": client})


def note_dict(note) -> dict:
    """One struct from the notes array, as a ClinicalNote-shaped dict."""
    values = dict(note)
    # Date and timestamp fields arrive as date/datetime/Timestamp objects; ClinicalNote wants 'YYYY-MM-DD'.
    if values.get("date") is not None and not isinstance(values["date"], str):
        values["date"] = pd.Timestamp(values["date"]).strftime("%Y-%m-%d")
    return values


@transform.using(
    notes=Input(NOTES_DATASET_RID),
    model=OpenAiGptChatLanguageModelInput(MODEL_RID),
    output=Output(OUTPUT_DATASET_RID),
)
def extract_cases(notes: LightweightInput, model: OpenAiGptChatLanguageModel, output: LightweightOutput):
    notes_df = notes.dataframe()

    client = httpx.Client(
        transport=FoundryModelsTransport(
            model,
            to_request=to_foundry_request,
            to_response=to_openai_response,
        )
    )
    rows = []
    try:
        agent = OrchestratorAgent(config=build_config(client))
        for mspi, patient_notes in zip(notes_df["mspi"], notes_df["notes"]):
            try:
                # Each cell is an array of struct dicts, or None when null. Not
                # `patient_notes or []`: an array of several notes has no truth value.
                raw_notes = [] if patient_notes is None else [note_dict(note) for note in patient_notes]
                result = agent.run(raw_notes, progress=False, capture_llm_content=False)
            except ValueError as error:
                # Empty, malformed, or duplicate-ID notes, rejected before any model call
                rows.append((mspi, "invalid_input", str(error), None))
            except OrchestratorRunError as error:
                rows.append((mspi, "failed", str(error), error.failure.model_dump_json()))
            else:
                rows.append((mspi, "completed", None, result.model_dump_json()))
    finally:
        client.close()

    output_df = pd.DataFrame(rows, columns=["mspi", *STRING_COLUMNS])
    output.write_table(output_df.astype(STRING_COLUMNS))

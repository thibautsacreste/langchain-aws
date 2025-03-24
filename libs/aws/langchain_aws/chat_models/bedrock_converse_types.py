"""
Pydantic types conforming to the Bedrock Converse API
https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html
"""

from typing import Annotated, Any, Literal

from annotated_types import Len
from pydantic import BaseModel, ConfigDict, Field, computed_field
from pydantic.alias_generators import to_camel


class BedrockBaseModel(BaseModel):
    """
    Base model for all Bedrock API models, which upon serialization:
      - excludes all unset optional fields
      - converts all field names to camelCase
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    def to_bedrock_dict(self):
        return self.model_dump(by_alias=True, exclude_unset=True)


class GuardrailConfiguration(BedrockBaseModel):
    guardrailIdentifier: str = Field(
        max_length=2048,
        pattern=r"^(([a-z0-9]+)|(arn:aws(-[^:]+)?:bedrock:[a-z0-9-]{1,20}:[0-9]{12}:guardrail/[a-z0-9]+))$",
    )
    guardrailVersion: str = Field(pattern=r"^(([1-9][0-9]{0,7})|(DRAFT))$")
    trace: Literal["enabled", "disabled"] | None = None


class InferenceConfiguration(BedrockBaseModel):
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    temperature: float | None = None
    top_p: float | None = None


class AnyToolChoice(BedrockBaseModel):
    any: dict = {}


class AutoToolChoice(BedrockBaseModel):
    auto: dict = {}


class SpecificToolChoice(BedrockBaseModel):
    class NamedTool(BedrockBaseModel):
        name: str

    tool: NamedTool


ToolChoice = AnyToolChoice | AutoToolChoice | SpecificToolChoice


class ToolInputSchema(BedrockBaseModel):
    json: dict[str, Any]


class ToolSpecification(BedrockBaseModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    description: str | None = None
    input_schema: ToolInputSchema


class Tool(BedrockBaseModel):
    tool_spec: ToolSpecification


class ToolConfiguration(BedrockBaseModel):
    tools: dict[str, Tool]
    tool_choice: ToolChoice | None = None


class Source(BedrockBaseModel):
    bytes: bytes


DocumentSource = Source
ImageSource = Source


class S3Location(BedrockBaseModel):
    uri: str = Field(
        min_length=1,
        max_length=1024,
        pattern=r"^s3://[a-z0-9][\.\-a-z0-9]{1,61}[a-z0-9](/.*)?$",
    )
    bucket_owner: str | None = Field(
        pattern=r"^[0-9]{12}$",
        max_length=12,
        default=None,
    )


class VideoSourceS3Location(BedrockBaseModel):
    s3_location: S3Location


VideoSource = Source | VideoSourceS3Location


class GuardrailConverseImageBlock(BedrockBaseModel):
    format: Literal["png", "jpeg"]
    source: ImageSource


class GuardrailConverseTextBlock(BedrockBaseModel):
    text: str
    qualifiers: list[
        Literal["grounding_source", "query", "guard_content"]
    ] | None = None


GuardrailConverseContentBlock = GuardrailConverseImageBlock | GuardrailConverseTextBlock


class DocumentBlock(BedrockBaseModel):
    format: Literal["pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"]
    name: str = Field(min_length=1, max_length=200)
    source: DocumentSource


class ImageBlock(BedrockBaseModel):
    format: Literal["png", "jpeg", "gif", "webp"]
    source: ImageSource


class VideoBlock(BedrockBaseModel):
    format: Literal[
        "mkv",
        "mov",
        "mp4",
        "webm",
        "flv",
        "mpeg",
        "mpg",
        "wmv",
        "three_gp",
    ]
    source: VideoSource


class ReasoningTextBlock(BedrockBaseModel):
    text: str
    signature: str | None = None


class ReasoningContentBlock(BedrockBaseModel):
    reasoning_text: ReasoningTextBlock | None = None
    redacted_content: bytes | None = None


class ToolUseBlock(BedrockBaseModel):
    input: dict[str, Any]
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    tool_use_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")


class ContentBlockDocument(BedrockBaseModel):
    document: DocumentBlock


class ContentBlockImage(BedrockBaseModel):
    image: ImageBlock


class ContentBlockJson(BedrockBaseModel):
    json: dict[str, Any]


class ContentBlockText(BedrockBaseModel):
    text: str | None = Field(min_length=1, default=None)


class ContentBlockVideo(BedrockBaseModel):
    video: VideoBlock


ToolResultContentBlock = (
    ContentBlockDocument
    | ContentBlockImage
    | ContentBlockJson
    | ContentBlockText
    | ContentBlockVideo
)


class ToolResultBlock(BedrockBaseModel):
    content: list[ToolResultContentBlock]
    tool_use_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    status: Literal["success", "error"] | None = None


class ContentBlockGuardrail(BedrockBaseModel):
    guard_content: GuardrailConverseContentBlock


class ContentBlockReasoning(BedrockBaseModel):
    reasoning_content: ReasoningContentBlock


class ContentBlockToolResult(BedrockBaseModel):
    tool_result: ToolResultBlock


class ContentBlockToolUse(BedrockBaseModel):
    tool_use: ToolUseBlock


class ContentBlockVideo(BedrockBaseModel):
    video: VideoBlock


ContentBlock = (
    ContentBlockDocument
    | ContentBlockGuardrail
    | ContentBlockImage
    | ContentBlockReasoning
    | ContentBlockText
    | ContentBlockToolResult
    | ContentBlockToolUse
    | ContentBlockVideo
)


class Message(BedrockBaseModel):
    """
    https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Message.html
    """

    content: list[ContentBlock]
    role: Literal["user", "assistant"]


class PerformanceConfiguration(BedrockBaseModel):
    latency: Literal["standard", "optimized"] | None = None


class PromptVariableValues(BedrockBaseModel):
    text: str | None = None


SystemContentBlock = ContentBlockText | ContentBlockGuardrail


class BedrockConverseParams(BedrockBaseModel):
    FieldPath = Annotated[str, Field(min_length=1, max_length=256)]
    MetadataKey = Annotated[
        str,
        Field(
            min_length=1, max_length=256, pattern=r"^[a-zA-Z0-9\s:_@$#=/+,-.]{1,256}$"
        ),
    ]
    MetadataValue = Annotated[
        str,
        Field(max_length=256, pattern=r"^[a-zA-Z0-9\s:_@$#=/+,-.]{0,256}$"),
    ]

    model_id: str = Field(min_length=1, max_length=2048)
    messages: list[Message] | None
    system: SystemContentBlock | None
    inference_config: InferenceConfiguration | None
    tool_config: ToolConfiguration | None
    guardrail_config: GuardrailConfiguration | None
    additional_model_request_fields: dict[str, Any] | None
    prompt_variables: dict[str, PromptVariableValues] | None
    additional_model_response_field_paths: Annotated[
        list[FieldPath], Len(max_length=10)
    ] | None
    request_metadata: dict[MetadataKey, MetadataValue] | None
    performance_config: PerformanceConfiguration | None

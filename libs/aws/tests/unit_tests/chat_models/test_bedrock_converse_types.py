from langchain_aws.chat_models.bedrock_converse_types import (
    BedrockConverseRequest,
    ContentBlockDocument,
    ContentBlockGuardrail,
    ContentBlockImage,
    ContentBlockJson,
    ContentBlockText,
    ContentBlockToolResult,
    ContentBlockToolUse,
    ContentBlockVideo,
    DocumentBlock,
    DocumentSource,
    GuardrailConfiguration,
    GuardrailConverseImageBlock,
    GuardrailConverseTextBlock,
    ImageBlock,
    ImageSource,
    InferenceConfiguration,
    Message,
    NamedTool,
    S3Location,
    SpecificToolChoice,
    Tool,
    ToolConfiguration,
    ToolInputSchema,
    ToolResultBlock,
    ToolSpecification,
    ToolUseBlock,
    VideoBlock,
    VideoSourceS3Location,
)


def test_guardrail_configuration():
    guardrail_arn = "arn:aws:bedrock:us-east-1:123456789012:guardrail/guardrail123"
    guardrail_config = GuardrailConfiguration(
        guardrail_identifier=guardrail_arn,
        guardrailVersion="1",
    )

    expected = {
        "guardrailIdentifier": guardrail_arn,
        "guardrailVersion": "1",
    }

    assert guardrail_config.to_bedrock_dict() == expected

    guardrail_config.trace = "enabled"

    assert guardrail_config.to_bedrock_dict()["trace"] == "enabled"


def test_inference_configuration():
    inference_config = InferenceConfiguration()

    assert inference_config.to_bedrock_dict() == {}

    inference_config = InferenceConfiguration(
        max_tokens=5,
        stop_sequences=["stop"],
        temperature=0.5,
        top_p=0.5,
    )

    expected = {
        "maxTokens": 5,
        "stopSequences": ["stop"],
        "temperature": 0.5,
        "topP": 0.5,
    }

    assert inference_config.to_bedrock_dict() == expected


def test_tool_configuration():
    get_weather_schema = {
        "type": "object",
        "properties": {
            "city_name": {"type": "string"},
        },
        "required": ["city_name"],
    }
    get_weather_tool = Tool(
        tool_spec=ToolSpecification(
            name="get_weather",
            input_schema=ToolInputSchema(
                json=get_weather_schema,
            ),
        ),
    )

    tool_config = ToolConfiguration(tools={"get_weather": get_weather_tool})

    expected = {
        "tools": {
            "get_weather": {
                "toolSpec": {
                    "name": "get_weather",
                    "inputSchema": {
                        "json": get_weather_schema,
                    },
                },
            },
        },
    }

    assert tool_config.to_bedrock_dict() == expected

    tool_config.tool_choice = SpecificToolChoice(tool=NamedTool(name="get_weather"))

    assert tool_config.to_bedrock_dict()["toolChoice"] == {
        "tool": {
            "name": "get_weather",
        },
    }


def test_content_blocks():
    text_block = ContentBlockText(
        text="some text",
    )

    text_block_expected = {
        "text": "some text",
    }

    assert text_block.to_bedrock_dict() == text_block_expected

    document_block = ContentBlockDocument(
        document=DocumentBlock(
            format="pdf",
            name="document.pdf",
            source=DocumentSource(
                bytes=b"pdf bytes",
            ),
        ),
    )

    document_block_expected = {
        "document": {
            "format": "pdf",
            "name": "document.pdf",
            "source": {
                "bytes": b"pdf bytes",
            },
        },
    }

    assert document_block.to_bedrock_dict() == document_block_expected

    guardrail_image_block = ContentBlockGuardrail(
        guard_content=GuardrailConverseImageBlock(
            format="png",
            source=ImageSource(
                bytes=b"png bytes",
            ),
        )
    )

    guardrail_image_block_expected = {
        "guardContent": {
            "format": "png",
            "source": {
                "bytes": b"png bytes",
            },
        },
    }

    assert guardrail_image_block.to_bedrock_dict() == guardrail_image_block_expected

    guardrail_text_block = ContentBlockGuardrail(
        guard_content=GuardrailConverseTextBlock(
            text="some very rude things",
            qualifiers=["grounding_source", "query", "guard_content"],
        )
    )

    guardrail_text_block_expected = {
        "guardContent": {
            "text": "some very rude things",
            "qualifiers": ["grounding_source", "query", "guard_content"],
        },
    }

    assert guardrail_text_block.to_bedrock_dict() == guardrail_text_block_expected

    image_block = ContentBlockImage(
        image=ImageBlock(
            format="gif",
            source=ImageSource(
                bytes=b"gif bytes",
            ),
        )
    )

    image_block_expected = {
        "image": {
            "format": "gif",
            "source": {
                "bytes": b"gif bytes",
            },
        },
    }

    assert image_block.to_bedrock_dict() == image_block_expected

    video_block = ContentBlockVideo(
        video=VideoBlock(
            format="mp4",
            source=VideoSourceS3Location(
                s3_location=S3Location(
                    uri="s3://bucket/key",
                    bucket_owner="123456789012",
                ),
            ),
        )
    )

    video_block_expected = {
        "video": {
            "format": "mp4",
            "source": {
                "s3Location": {
                    "uri": "s3://bucket/key",
                    "bucketOwner": "123456789012",
                },
            },
        },
    }

    assert video_block.to_bedrock_dict() == video_block_expected

    tool_use_block = ContentBlockToolUse(
        tool_use=ToolUseBlock(
            name="get_weather",
            input={"city_name": "London"},
            tool_use_id="123",
        )
    )

    tool_use_block_expected = {
        "toolUse": {
            "name": "get_weather",
            "input": {"city_name": "London"},
            "toolUseId": "123",
        },
    }

    assert tool_use_block.to_bedrock_dict() == tool_use_block_expected

    json_block = ContentBlockJson(
        json={"my_key": "my_value"},
    )

    json_block_expected = {
        "json": {"my_key": "my_value"},
    }

    assert json_block.to_bedrock_dict() == json_block_expected

    tool_result_block = ContentBlockToolResult(
        tool_result=ToolResultBlock(
            content=[json_block, text_block, image_block, video_block],
            tool_use_id="123",
            status="success",
        )
    )

    tool_result_block_expected = {
        "toolResult": {
            "content": [
                json_block_expected,
                text_block_expected,
                image_block_expected,
                video_block_expected,
            ],
            "toolUseId": "123",
            "status": "success",
        },
    }

    assert tool_result_block.to_bedrock_dict() == tool_result_block_expected


def test_messages():
    messages = [
        Message(
            role="user",
            content=[
                ContentBlockText(
                    text="Hello, how are you?",
                ),
            ],
        ),
        Message(
            role="assistant",
            content=[
                ContentBlockText(
                    text="I'm doing well, thank you.",
                ),
            ],
        ),
    ]

    messages_expected = [
        {
            "role": "user",
            "content": [
                {
                    "text": "Hello, how are you?",
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "text": "I'm doing well, thank you.",
                },
            ],
        },
    ]

    assert [msg.to_bedrock_dict() for msg in messages] == messages_expected


def test_bedrock_converse_snake():
    camel_kwargs = {
        "modelId": "model_id",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "guardContent": {
                            "text": "Hello, how are you?",
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "text": "This is a tough question!",
                                "signature": "token",
                            },
                        },
                    }
                ],
            },
        ],
        "system": {
            "text": "system text",
        },
        "inferenceConfig": {
            "maxTokens": 5,
            "stopSequences": ["stop"],
            "temperature": 0.5,
            "topP": 0.5,
        },
        "toolConfig": {
            "tools": {
                "get_weather": {
                    "toolSpec": {
                        "name": "get_weather",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "city_name": {"type": "string"},
                                },
                                "required": ["city_name"],
                            },
                        },
                    },
                },
            },
            "toolChoice": {
                "tool": {
                    "name": "get_weather",
                },
            },
        },
        "guardrailConfig": {
            "guardrailIdentifier": "arn:aws:bedrock:us-east-1:123456789012:guardrail/guardrail123",
            "guardrailVersion": "1",
        },
        "additionalModelRequestFields": {
            "my_key": "my_value",
        },
    }

    snake_kwargs = {
        "model_id": "model_id",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "guard_content": {
                            "text": "Hello, how are you?",
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "reasoning_content": {
                            "reasoning_text": {
                                "text": "This is a tough question!",
                                "signature": "token",
                            },
                        }
                    },
                ],
            },
        ],
        "system": {
            "text": "system text",
        },
        "inference_config": {
            "max_tokens": 5,
            "stop_sequences": ["stop"],
            "temperature": 0.5,
            "top_p": 0.5,
        },
        "tool_config": {
            "tools": {
                "get_weather": {
                    "tool_spec": {
                        "name": "get_weather",
                        "input_schema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "city_name": {"type": "string"},
                                },
                                "required": ["city_name"],
                            },
                        },
                    },
                },
            },
            "tool_choice": {
                "tool": {
                    "name": "get_weather",
                },
            },
        },
        "guardrail_config": {
            "guardrail_identifier": "arn:aws:bedrock:us-east-1:123456789012:guardrail/guardrail123",
            "guardrail_version": "1",
        },
        "additional_model_request_fields": {
            "my_key": "my_value",
        },
    }

    assert BedrockConverseRequest(**snake_kwargs).to_bedrock_dict() == camel_kwargs
    assert BedrockConverseRequest(**camel_kwargs).to_bedrock_dict() == camel_kwargs

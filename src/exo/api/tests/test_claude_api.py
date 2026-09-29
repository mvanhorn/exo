"""Tests for Claude Messages API conversion functions and types."""

import pydantic
import pytest

from exo.api.adapters.claude import (
    claude_request_to_text_generation,
    finish_reason_to_claude_stop_reason,
)
from exo.api.types.claude_api import (
    ClaudeMessage,
    ClaudeMessagesRequest,
    ClaudeMessagesResponse,
    ClaudeMessageStart,
    ClaudeTextBlock,
    ClaudeUsage,
)
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import ChatTemplateValue, InputMessage


def _input_pairs(messages: list[InputMessage]) -> list[tuple[str, str]]:
    return [(str(message.role), str(message.content)) for message in messages]


def _template_pairs(
    messages: list[dict[str, ChatTemplateValue]] | None,
) -> list[tuple[str, str]]:
    assert messages is not None
    pairs: list[tuple[str, str]] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        assert isinstance(role, str)
        assert isinstance(content, str)
        pairs.append((role, content))
    return pairs


class TestFinishReasonToClaudeStopReason:
    """Tests for finish_reason to Claude stop_reason mapping."""

    def test_stop_maps_to_end_turn(self):
        assert finish_reason_to_claude_stop_reason("stop") == "end_turn"

    def test_length_maps_to_max_tokens(self):
        assert finish_reason_to_claude_stop_reason("length") == "max_tokens"

    def test_tool_calls_maps_to_tool_use(self):
        assert finish_reason_to_claude_stop_reason("tool_calls") == "tool_use"

    def test_function_call_maps_to_tool_use(self):
        assert finish_reason_to_claude_stop_reason("function_call") == "tool_use"

    def test_content_filter_maps_to_end_turn(self):
        assert finish_reason_to_claude_stop_reason("content_filter") == "end_turn"

    def test_none_returns_none(self):
        assert finish_reason_to_claude_stop_reason(None) is None


class TestClaudeRequestToInternal:
    """Tests for converting Claude Messages API requests to TextGenerationTaskParams."""

    async def test_basic_request_conversion(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(role="user", content="Hello"),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.model == "claude-3-opus"
        assert params.max_output_tokens == 100
        assert isinstance(params.input, list)
        assert len(params.input) == 1
        assert params.input[0].role == "user"
        assert params.input[0].content == "Hello"
        assert params.instructions is None

    async def test_request_with_system_string(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            system="You are a helpful assistant.",
            messages=[
                ClaudeMessage(role="user", content="Hello"),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.instructions == "You are a helpful assistant."
        assert isinstance(params.input, list)
        assert len(params.input) == 1
        assert params.input[0].role == "user"
        assert params.input[0].content == "Hello"

    async def test_request_with_system_text_blocks(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            system=[
                ClaudeTextBlock(text="You are helpful. "),
                ClaudeTextBlock(text="Be concise."),
            ],
            messages=[
                ClaudeMessage(role="user", content="Hello"),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.instructions == "You are helpful. Be concise."
        assert isinstance(params.input, list)
        assert len(params.input) == 1

    async def test_request_with_content_blocks(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeTextBlock(text="First part. "),
                        ClaudeTextBlock(text="Second part."),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert isinstance(params.input, list)
        assert len(params.input) == 1
        assert params.input[0].content == "First part. Second part."

    async def test_request_with_multi_turn_conversation(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(role="user", content="Hello"),
                ClaudeMessage(role="assistant", content="Hi there!"),
                ClaudeMessage(role="user", content="How are you?"),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert isinstance(params.input, list)
        assert len(params.input) == 3
        assert params.input[0].role == "user"
        assert params.input[1].role == "assistant"
        assert params.input[2].role == "user"

    async def test_request_with_optional_parameters(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[ClaudeMessage(role="user", content="Hello")],
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            stop_sequences=["STOP", "END"],
            stream=True,
        )
        params = await claude_request_to_text_generation(request)

        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 40
        assert params.stop == ["STOP", "END"]
        assert params.stream is True


class TestClaudeMessagesRequestValidation:
    """Tests for Claude Messages API request validation."""

    def test_request_requires_model(self):
        with pytest.raises(pydantic.ValidationError):
            ClaudeMessagesRequest.model_validate(
                {
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            )

    def test_request_requires_max_tokens(self):
        with pytest.raises(pydantic.ValidationError):
            ClaudeMessagesRequest.model_validate(
                {
                    "model": "claude-3-opus",
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            )

    def test_request_requires_messages(self):
        with pytest.raises(pydantic.ValidationError):
            ClaudeMessagesRequest.model_validate(
                {
                    "model": "claude-3-opus",
                    "max_tokens": 100,
                }
            )

    def test_raw_request_accepts_system_role_at_messages_index_one(self):
        """Xcode places a system turn inside messages; index 1 must validate."""
        request = ClaudeMessagesRequest.model_validate(
            {
                "model": "claude-3-opus",
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "system", "content": "Follow the project rules."},
                ],
            }
        )

        assert request.messages[1].role == "system"
        assert request.messages[1].content == "Follow the project rules."

    @pytest.mark.parametrize("role", ["developer", "tool", "function", "System"])
    def test_unsupported_message_roles_rejected(self, role: str):
        with pytest.raises(pydantic.ValidationError):
            ClaudeMessagesRequest.model_validate(
                {
                    "model": "claude-3-opus",
                    "max_tokens": 100,
                    "messages": [{"role": role, "content": "Hello"}],
                }
            )

    def test_response_roles_remain_assistant_only(self):
        response = ClaudeMessagesResponse.model_validate(
            {
                "id": "msg_1",
                "model": "claude-3-opus",
                "content": [{"type": "text", "text": "Hi"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
        assert response.role == "assistant"

        start = ClaudeMessageStart.model_validate(
            {
                "id": "msg_1",
                "model": "claude-3-opus",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        )
        assert start.role == "assistant"

        explicit = ClaudeMessagesResponse(
            id="msg_explicit",
            role="assistant",
            model="claude-3-opus",
            content=[ClaudeTextBlock(text="Hi")],
            usage=ClaudeUsage(input_tokens=1, output_tokens=1),
        )
        assert explicit.role == "assistant"

        for role in ("user", "system"):
            with pytest.raises(pydantic.ValidationError):
                ClaudeMessagesResponse.model_validate(
                    {
                        "id": "msg_1",
                        "role": role,
                        "model": "claude-3-opus",
                        "content": [{"type": "text", "text": "Hi"}],
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                )
            with pytest.raises(pydantic.ValidationError):
                ClaudeMessageStart.model_validate(
                    {
                        "id": "msg_1",
                        "role": role,
                        "model": "claude-3-opus",
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    }
                )


class TestInlineSystemMessageConversion:
    """Inline system turns stay in place beside the top-level system prompt."""

    @pytest.mark.parametrize(
        ("system_content", "expected"),
        [
            pytest.param("Be concise.", "Be concise.", id="string"),
            pytest.param(
                [{"type": "text", "text": "Only one."}],
                "Only one.",
                id="single-text-block",
            ),
            pytest.param(
                [
                    {"type": "text", "text": "A"},
                    {"type": "text", "text": "B"},
                    {"type": "text", "text": "C"},
                ],
                "ABC",
                id="multiple-text-blocks",
            ),
        ],
    )
    async def test_inline_system_content_preserves_role_and_order(
        self,
        system_content: str | list[dict[str, str]],
        expected: str,
    ):
        request = ClaudeMessagesRequest.model_validate(
            {
                "model": "claude-3-opus",
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "system", "content": system_content},
                    {"role": "assistant", "content": "Hi there"},
                ],
            }
        )
        params = await claude_request_to_text_generation(request)

        expected_pairs = [
            ("user", "Hello"),
            ("system", expected),
            ("assistant", "Hi there"),
        ]
        assert _input_pairs(params.input) == expected_pairs
        assert _template_pairs(params.chat_template_messages) == expected_pairs
        assert params.instructions is None

    @pytest.mark.parametrize(
        ("messages", "expected"),
        [
            pytest.param(
                [
                    {"role": "system", "content": "Start"},
                    {"role": "user", "content": "Hello"},
                ],
                [("system", "Start"), ("user", "Hello")],
                id="beginning",
            ),
            pytest.param(
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "system", "content": "Middle"},
                    {"role": "assistant", "content": "Hi there"},
                ],
                [
                    ("user", "Hello"),
                    ("system", "Middle"),
                    ("assistant", "Hi there"),
                ],
                id="middle",
            ),
            pytest.param(
                [
                    {"role": "system", "content": "One"},
                    {"role": "user", "content": "Hello"},
                    {"role": "system", "content": "Two"},
                    {"role": "assistant", "content": "Hi there"},
                ],
                [
                    ("system", "One"),
                    ("user", "Hello"),
                    ("system", "Two"),
                    ("assistant", "Hi there"),
                ],
                id="repeated",
            ),
            pytest.param(
                [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": "Hello"},
                ],
                [("system", ""), ("user", "Hello")],
                id="empty-string-at-start",
            ),
            pytest.param(
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "system", "content": ""},
                    {"role": "assistant", "content": "Hi there"},
                ],
                [
                    ("user", "Hello"),
                    ("system", ""),
                    ("assistant", "Hi there"),
                ],
                id="empty-string-in-middle",
            ),
            pytest.param(
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": ""}],
                    },
                    {"role": "user", "content": "Hello"},
                ],
                [("system", ""), ("user", "Hello")],
                id="empty-text-block",
            ),
        ],
    )
    async def test_inline_system_position_repetition_and_empty_text(
        self,
        messages: list[dict[str, str | list[dict[str, str]]]],
        expected: list[tuple[str, str]],
    ):
        request = ClaudeMessagesRequest.model_validate(
            {
                "model": "claude-3-opus",
                "max_tokens": 100,
                "messages": messages,
            }
        )
        params = await claude_request_to_text_generation(request)

        assert _input_pairs(params.input) == expected
        assert _template_pairs(params.chat_template_messages) == expected
        assert params.instructions is None

    @pytest.mark.parametrize(
        ("system", "expected_instructions"),
        [
            pytest.param(
                "Top-level instructions.",
                "Top-level instructions.",
                id="string",
            ),
            pytest.param(
                [
                    {"type": "text", "text": "Top "},
                    {"type": "text", "text": "level."},
                ],
                "Top level.",
                id="text-blocks",
            ),
            pytest.param(
                "x-anthropic-billing-header: cc_version=1.0;\nKeep this.",
                "Keep this.",
                id="string-telemetry",
            ),
            pytest.param(
                [
                    {
                        "type": "text",
                        "text": "x-anthropic-billing-header: cc_version=1.0;\n",
                    },
                    {"type": "text", "text": "Keep this."},
                ],
                "Keep this.",
                id="text-block-telemetry",
            ),
        ],
    )
    async def test_toplevel_system_coexists_with_inline_system(
        self,
        system: str | list[dict[str, str]],
        expected_instructions: str,
    ):
        inline = "x-anthropic-billing-header: cc_version=9;\nInline rule."
        request = ClaudeMessagesRequest.model_validate(
            {
                "model": "claude-3-opus",
                "max_tokens": 100,
                "system": system,
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "system", "content": inline},
                    {"role": "assistant", "content": "Hi there"},
                ],
            }
        )
        params = await claude_request_to_text_generation(request)

        assert params.instructions == expected_instructions
        template_pairs = _template_pairs(params.chat_template_messages)
        assert template_pairs[0] == ("system", expected_instructions)
        assert template_pairs.count(("system", inline)) == 1
        assert template_pairs == [
            ("system", expected_instructions),
            ("user", "Hello"),
            ("system", inline),
            ("assistant", "Hi there"),
        ]
        assert _input_pairs(params.input) == [
            ("user", "Hello"),
            ("system", inline),
            ("assistant", "Hi there"),
        ]

    @pytest.mark.parametrize("stream", [True, False])
    async def test_stream_flag_retained_with_inline_system(self, stream: bool):
        request = ClaudeMessagesRequest.model_validate(
            {
                "model": "claude-3-opus",
                "max_tokens": 100,
                "stream": stream,
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "system", "content": "Follow the rules."},
                ],
            }
        )
        params = await claude_request_to_text_generation(request)

        assert params.stream is stream
        assert _input_pairs(params.input) == [
            ("user", "Hello"),
            ("system", "Follow the rules."),
        ]

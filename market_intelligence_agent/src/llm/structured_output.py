"""
Structured Output Helpers

Converts Pydantic schemas to LLM tool definitions (Anthropic/OpenAI formats).
Validates LLM responses against schemas.

Why structured outputs?
- Ensures LLM returns data in expected format
- Type safety (Pydantic validation)
- No parsing errors (no "extract JSON from markdown")

Design:
- Use Anthropic's "tool use" feature for Claude
- Use OpenAI's "function calling" for GPT-4
- Both map to Pydantic models
"""

from typing import Type, Dict, Any, List, Optional
from pydantic import BaseModel
import json


def pydantic_to_anthropic_tool(
    model: Type[BaseModel],
    name: str,
    description: str
) -> Dict[str, Any]:
    """
    Convert a Pydantic model to Anthropic tool definition.

    Anthropic's tool use format:
    {
        "name": "extract_event",
        "description": "Extract a market signal event",
        "input_schema": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }

    Args:
        model: Pydantic model class (e.g., MarketSignalEvent)
        name: Tool name (e.g., "extract_market_signal_event")
        description: What the tool does

    Returns:
        Anthropic tool definition dict

    Example:
        tool = pydantic_to_anthropic_tool(
            MarketSignalEvent,
            "extract_market_signal_event",
            "Extract structured market signal event from content"
        )
    """
    # Get JSON schema from Pydantic model
    # mode='validation' ensures we get the input schema, not serialization
    schema = model.model_json_schema(mode='validation')

    # Anthropic tool format
    tool_def = {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
            "$defs": schema.get("$defs", {})  # Include nested definitions
        }
    }

    # Debug: print schema for troubleshooting
    # import json
    # print(f"\n[DEBUG] Tool schema for {name}:")
    # print(json.dumps(tool_def['input_schema'], indent=2)[:500])

    return tool_def


def pydantic_to_openai_function(
    model: Type[BaseModel],
    name: str,
    description: str
) -> Dict[str, Any]:
    """
    Convert a Pydantic model to OpenAI function definition.

    OpenAI's function calling format:
    {
        "type": "function",
        "function": {
            "name": "extract_event",
            "description": "Extract a market signal event",
            "parameters": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }
    }

    Args:
        model: Pydantic model class
        name: Function name
        description: What the function does

    Returns:
        OpenAI function definition dict
    """
    # Get JSON schema from Pydantic model
    schema = model.model_json_schema()

    # OpenAI function format
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        }
    }


def extract_structured_output(
    response: Dict[str, Any],
    model: Type[BaseModel]
) -> BaseModel:
    """
    Extract and validate structured output from LLM response.

    Works with both Anthropic and OpenAI responses (standardized format
    from LLMProvider).

    Args:
        response: Response from LLMProvider.generate()
        model: Pydantic model to validate against

    Returns:
        Instance of the Pydantic model

    Raises:
        ValueError: If no tool calls found or validation fails

    Example:
        response = llm_provider.generate(
            messages=[...],
            tools=[pydantic_to_anthropic_tool(MarketSignalEvent, ...)]
        )
        event = extract_structured_output(response, MarketSignalEvent)
    """
    # Check if response has tool calls
    if not response.get('tool_calls'):
        raise ValueError("No tool calls in response - LLM may not have used the tool")

    # Get first tool call (we typically only use one)
    tool_call = response['tool_calls'][0]

    # Extract input parameters
    input_data = tool_call['input']

    # Validate against Pydantic model
    try:
        return model.model_validate(input_data)
    except Exception as e:
        raise ValueError(f"Failed to validate LLM output against schema: {e}")


def create_tool_choice(tool_name: str, provider: str) -> Dict[str, Any]:
    """
    Create tool choice specification to force LLM to use a specific tool.

    Why force tool use?
    - Ensures LLM returns structured output (not just text)
    - Prevents LLM from saying "I can't do that" instead of calling tool

    Args:
        tool_name: Name of tool to force
        provider: "anthropic" or "openai"

    Returns:
        Tool choice dict in provider-specific format

    Example:
        tool_choice = create_tool_choice("extract_market_signal_event", "anthropic")
    """
    if provider == "anthropic":
        # Anthropic format
        return {
            "type": "tool",
            "name": tool_name
        }
    elif provider == "openai":
        # OpenAI format
        return {
            "type": "function",
            "function": {"name": tool_name}
        }
    else:
        raise ValueError(f"Unknown provider: {provider}")


def validate_partial_output(
    data: Dict[str, Any],
    model: Type[BaseModel],
    required_fields: Optional[List[str]] = None
) -> tuple[bool, List[str]]:
    """
    Validate partial output (e.g., during multi-stage extraction).

    Useful when building up a structured object across multiple LLM calls.

    Args:
        data: Partial data dict
        model: Pydantic model to validate against
        required_fields: Specific fields that must be present

    Returns:
        (is_valid, missing_fields)

    Example:
        # Stage 1: Extract what_changed and why_it_matters
        stage1_data = {"what_changed": "...", "why_it_matters": "..."}
        is_valid, missing = validate_partial_output(
            stage1_data,
            MarketSignalEvent,
            required_fields=["what_changed", "why_it_matters"]
        )
    """
    missing_fields = []

    # If specific fields specified, check those
    if required_fields:
        for field in required_fields:
            if field not in data or data[field] is None:
                missing_fields.append(field)
    else:
        # Otherwise, check all required fields from schema
        schema = model.model_json_schema()
        required = schema.get('required', [])

        for field in required:
            if field not in data or data[field] is None:
                missing_fields.append(field)

    return (len(missing_fields) == 0, missing_fields)


# Convenience function for common use case
def generate_with_structured_output(
    llm_provider,
    messages: List[Dict[str, str]],
    output_model: Type[BaseModel],
    tool_name: str,
    tool_description: str,
    task_complexity: str = "complex",
    temperature: Optional[float] = None
) -> BaseModel:
    """
    One-liner for generating structured output.

    Handles:
    1. Converting Pydantic model to tool definition
    2. Forcing tool use
    3. Calling LLM
    4. Extracting and validating output

    Args:
        llm_provider: LLMProvider instance
        messages: Messages to send to LLM
        output_model: Pydantic model for output
        tool_name: Name of tool
        tool_description: Description of what tool does
        task_complexity: "simple" or "complex"
        temperature: Optional temperature override

    Returns:
        Instance of output_model

    Example:
        event = generate_with_structured_output(
            llm_provider=llm,
            messages=[{"role": "user", "content": "Extract event from: ..."}],
            output_model=MarketSignalEvent,
            tool_name="extract_market_signal_event",
            tool_description="Extract structured market signal event",
            task_complexity="complex"
        )
    """
    # Determine provider based on task complexity
    provider = "anthropic" if task_complexity == "complex" else "openai"

    # Convert Pydantic model to tool definition
    if provider == "anthropic":
        tool = pydantic_to_anthropic_tool(output_model, tool_name, tool_description)
    else:
        tool = pydantic_to_openai_function(output_model, tool_name, tool_description)

    # Create tool choice (force tool use)
    tool_choice = create_tool_choice(tool_name, provider)

    # Generate response
    response = llm_provider.generate(
        messages=messages,
        task_complexity=task_complexity,
        temperature=temperature,
        tools=[tool],
        tool_choice=tool_choice
    )

    # Extract and validate
    return extract_structured_output(response, output_model)

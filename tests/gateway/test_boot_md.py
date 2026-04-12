from unittest.mock import Mock, patch

from gateway.builtin_hooks import boot_md


def test_run_boot_agent_uses_gateway_model_and_runtime():
    fake_agent = Mock()
    fake_agent.run_conversation.return_value = {"final_response": "[SILENT]"}

    with patch("gateway.run._load_gateway_config", return_value={"model": {"default": "MiniMax-M2.7"}}), \
         patch("gateway.run._resolve_gateway_model", return_value="MiniMax-M2.7"), \
         patch(
             "gateway.run._resolve_runtime_agent_kwargs",
             return_value={
                 "provider": "minimax",
                 "base_url": "https://api.minimax.io/anthropic",
                 "api_key": "test-key",
                 "api_mode": "anthropic_messages",
             },
         ), \
         patch("run_agent.AIAgent", return_value=fake_agent) as mock_agent:
        boot_md._run_boot_agent("# Startup Checklist\n\n1. Say hi")

    mock_agent.assert_called_once_with(
        model="MiniMax-M2.7",
        provider="minimax",
        base_url="https://api.minimax.io/anthropic",
        api_key="test-key",
        api_mode="anthropic_messages",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        max_iterations=20,
    )
    fake_agent.run_conversation.assert_called_once()

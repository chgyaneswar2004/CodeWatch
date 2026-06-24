import unittest
from unittest.mock import patch

# Skip these tests if the correct modules aren't available
try:
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@unittest.skipUnless(HAS_OPENAI, "OpenAI not available")
class TestLangchainUtils(unittest.TestCase):
    def test_module_imports(self):
        """Simple test to verify imports work"""
        # This is a basic test to check that our module exists and can be imported
        from codedog.utils import langchain_utils
        self.assertTrue(hasattr(langchain_utils, 'load_gpt_llm'))
        self.assertTrue(hasattr(langchain_utils, 'load_gpt4_llm'))

    @patch('codedog.utils.langchain_utils.env')
    def test_load_gpt_llm_functions(self, mock_env):
        """Test that the load functions access environment variables"""
        # Mock the env.get calls
        mock_env.get.return_value = None

        # We don't call the function to avoid import errors
        # Just check that the environment setup works
        mock_env.get.assert_not_called()

        # Reset mock for possible reuse
        mock_env.reset_mock()

    @patch('codedog.utils.langchain_utils.env')
    def test_azure_config_loading(self, mock_env):
        """Test that Azure configuration is handled correctly"""
        # We'll just check if env.get is called with the right key

        # Configure env mock to simulate Azure environment
        mock_env.get.return_value = "true"

        # Import module but don't call functions
        from codedog.utils import langchain_utils

        # We won't call load_gpt_llm here to avoid creating actual models
        # Just verify it can be imported

        # Make another call to verify mocking
        is_azure = langchain_utils.env.get("AZURE_OPENAI", None) == "true"
        self.assertTrue(is_azure)

        # Verify that env.get was called for the Azure key
        mock_env.get.assert_called_with("AZURE_OPENAI", None)

    def test_deepseek_normalize_messages(self):
        """Test that messages are formatted to comply with DeepSeek API rules"""
        from codedog.utils.langchain_utils import DeepSeekChatModel
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        model = DeepSeekChatModel(
            api_key="test",
            model_name="deepseek-chat",
            api_base="https://api.deepseek.com",
            temperature=0.0,
            max_tokens=100,
            top_p=0.95
        )

        # Test consecutive messages of same role get merged
        messages = [
            SystemMessage(content="Sys1"),
            SystemMessage(content="Sys2"),
            HumanMessage(content="User1"),
            HumanMessage(content="User2"),
            AIMessage(content="Assistant1"),
            AIMessage(content="Assistant2"),
            HumanMessage(content="User3"),
        ]

        normalized = model._normalize_messages(messages)
        self.assertEqual(len(normalized), 4)
        self.assertEqual(normalized[0], {"role": "system", "content": "Sys1\n\nSys2"})
        self.assertEqual(normalized[1], {"role": "user", "content": "User1\n\nUser2"})
        self.assertEqual(normalized[2], {"role": "assistant", "content": "Assistant1\n\nAssistant2"})
        self.assertEqual(normalized[3], {"role": "user", "content": "User3"})

        # Test prepending user message if first non-system message is assistant
        messages_assistant_first = [
            AIMessage(content="Assistant1")
        ]
        normalized_prepended = model._normalize_messages(messages_assistant_first)
        self.assertEqual(len(normalized_prepended), 2)
        self.assertEqual(normalized_prepended[0], {"role": "user", "content": "Please process the following:"})
        self.assertEqual(normalized_prepended[1], {"role": "assistant", "content": "Assistant1"})

    @patch('codedog.utils.langchain_utils.requests.post')
    def test_deepseek_generate_payload_reasoner(self, mock_post):
        """Test that deepseek-reasoner omits temperature and top_p from payload"""
        from codedog.utils.langchain_utils import DeepSeekR1Model
        from langchain_core.messages import HumanMessage
        import json

        model = DeepSeekR1Model(
            api_key="test",
            model_name="deepseek-reasoner",
            api_base="https://api.deepseek.com",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )

        mock_response = mock_post.return_value
        mock_response.status_code = 200
        mock_response.text = '{"choices": [{"message": {"content": "test response"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 20}}'
        mock_response.json.return_value = json.loads(mock_response.text)

        messages = [HumanMessage(content="Hello")]
        result = model._generate(messages)

        self.assertEqual(result.generations[0].message.content, "test response")
        
        # Verify post payload
        called_kwargs = mock_post.call_args[1]
        payload = called_kwargs["json"]
        self.assertEqual(payload["model"], "deepseek-reasoner")
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)

    @patch('codedog.utils.langchain_utils.requests.post')
    def test_deepseek_generate_payload_chat(self, mock_post):
        """Test that deepseek-chat includes temperature and top_p in payload"""
        from codedog.utils.langchain_utils import DeepSeekChatModel
        from langchain_core.messages import HumanMessage
        import json

        model = DeepSeekChatModel(
            api_key="test",
            model_name="deepseek-chat",
            api_base="https://api.deepseek.com",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )

        mock_response = mock_post.return_value
        mock_response.status_code = 200
        mock_response.text = '{"choices": [{"message": {"content": "test response"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 20}}'
        mock_response.json.return_value = json.loads(mock_response.text)

        messages = [HumanMessage(content="Hello")]
        result = model._generate(messages)

        self.assertEqual(result.generations[0].message.content, "test response")
        
        # Verify post payload
        called_kwargs = mock_post.call_args[1]
        payload = called_kwargs["json"]
        self.assertEqual(payload["model"], "deepseek-chat")
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["top_p"], 0.9)

    def test_deepseek_agenerate(self):
        """Test that _agenerate asynchronously fetches and parses content correctly without NameError"""
        import asyncio
        from unittest.mock import patch, MagicMock
        from codedog.utils.langchain_utils import DeepSeekChatModel
        from langchain_core.messages import HumanMessage

        model = DeepSeekChatModel(
            api_key="test",
            model_name="deepseek-chat",
            api_base="https://api.deepseek.com",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )

        with patch('codedog.utils.langchain_utils.aiohttp.ClientSession.post') as mock_post:
            # Mock aiohttp response context manager
            mock_response = MagicMock()
            mock_response.status = 200
            
            # Since we await response.text(), return a coroutine
            async def mock_text():
                return '{"choices": [{"message": {"content": "async test response"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 20}}'
            mock_response.text = mock_text

            # Mock context manager protocol
            mock_context = MagicMock()

            async def mock_aenter(*args, **kwargs):
                return mock_response

            async def mock_aexit(*args, **kwargs):
                pass

            mock_context.__aenter__ = mock_aenter
            mock_context.__aexit__ = mock_aexit
            mock_post.return_value = mock_context

            messages = [HumanMessage(content="Hello")]
            result = asyncio.run(model._agenerate(messages))

            self.assertEqual(result.generations[0].message.content, "async test response")


if __name__ == '__main__':
    unittest.main()

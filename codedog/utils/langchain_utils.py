from functools import lru_cache
from os import environ as env
from typing import Dict, Any, List, Optional
import inspect
import os
import time

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, ConfigDict
import requests
import aiohttp
import json
from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
import logging
import traceback
import asyncio

logger = logging.getLogger(__name__)


def log_error(e: Exception, message: str, response_text: str = None):
    """Log error with file name and line number"""
    frame = inspect.currentframe()
    # Get the caller's frame (1 level up)
    caller = frame.f_back
    if caller:
        file_name = os.path.basename(caller.f_code.co_filename)
        line_no = caller.f_lineno
        error_msg = f"{file_name}:{line_no} - {message}: {str(e)}"
        if response_text:
            error_msg += f"\nResponse: {response_text}"
        error_msg += f"\n{traceback.format_exc()}"
        logger.error(error_msg)
    else:
        error_msg = f"{message}: {str(e)}"
        if response_text:
            error_msg += f"\nResponse: {response_text}"
        error_msg += f"\n{traceback.format_exc()}"
        logger.error(error_msg)


# Define a custom class for DeepSeek model since it's not available in langchain directly
class DeepSeekChatModel(BaseChatModel):
    """DeepSeek Chat Model"""

    api_key: str
    model_name: str
    api_base: str
    temperature: float
    max_tokens: int
    top_p: float
    timeout: int = 600  # 增加默认超时时间到600秒
    max_retries: int = 3  # 最大重试次数
    retry_delay: int = 5  # 重试间隔（秒）
    total_tokens: int = 0
    total_cost: float = 0.0
    failed_requests: int = 0  # 失败请求计数

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on token usage and model type."""
        model_lower = self.model_name.lower() if self.model_name else ""
        # DeepSeek-R1 pricing: Input: $0.55/M, Output: $2.19/M
        # DeepSeek-V3 pricing: Input: $0.14/M, Output: $0.28/M
        if "r1" in model_lower or "reason" in model_lower:
            input_rate = 0.55 / 1_000_000
            output_rate = 2.19 / 1_000_000
        else:
            input_rate = 0.14 / 1_000_000
            output_rate = 0.28 / 1_000_000
            
        return (prompt_tokens * input_rate) + (completion_tokens * output_rate)

    def _normalize_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Format LangChain messages to conform to DeepSeek API constraints."""
        if not messages:
            return []

        # 1. Convert to dicts with roles and filter/merge system messages
        system_contents = []
        other_messages = []
        for message in messages:
            role = "user" if isinstance(message, HumanMessage) else "system" if isinstance(message, SystemMessage) else "assistant"
            content = message.content
            if isinstance(content, list):
                # Handle cases where message content is a list of blocks
                content_str = "\n".join([str(block) for block in content])
            else:
                content_str = str(content)
            
            content_str = content_str.strip()
            if not content_str:
                content_str = "..."

            if role == "system":
                system_contents.append(content_str)
            else:
                other_messages.append({"role": role, "content": content_str})

        # 2. Merge consecutive messages of the same role
        merged_others = []
        for msg in other_messages:
            if not merged_others:
                merged_others.append(msg)
            else:
                last_msg = merged_others[-1]
                if last_msg["role"] == msg["role"]:
                    last_msg["content"] = (last_msg["content"] + "\n\n" + msg["content"]).strip()
                else:
                    merged_others.append(msg)

        # 3. Ensure the sequence starts with user (if there are any other messages)
        if merged_others and merged_others[0]["role"] == "assistant":
            merged_others.insert(0, {"role": "user", "content": "Please process the following:"})

        # 4. Construct final message list
        final_messages = []
        if system_contents:
            final_messages.append({
                "role": "system",
                "content": "\n\n".join(system_contents).strip()
            })
        final_messages.extend(merged_others)
        
        return final_messages

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the DeepSeek API."""
        try:
            # Convert LangChain messages to DeepSeek format with normalization
            deepseek_messages = self._normalize_messages(messages)

            # Prepare API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            is_reasoner = "r1" in self.model_name.lower() or "reason" in self.model_name.lower() or self._llm_type == "deepseek-reasoner"
            
            payload = {
                "model": self.model_name,
                "messages": deepseek_messages,
                "max_tokens": self.max_tokens,
            }
            if not is_reasoner:
                payload["temperature"] = self.temperature
                payload["top_p"] = self.top_p
                if stop:
                    payload["stop"] = stop

            # Log request details for debugging
            logger.debug(f"DeepSeek API request to {self.api_base}")
            logger.debug(f"Model: {self.model_name}")
            logger.debug(f"Payload: {json.dumps(payload, ensure_ascii=False)}")

            # Ensure API base URL is properly formatted and construct endpoint
            api_base = self.api_base.rstrip('/')
            endpoint = f"{api_base}/v1/chat/completions"

            # Implement retry mechanism
            retries = 0
            last_error = None

            while retries < self.max_retries:
                try:
                    # Calculate current timeout using exponential backoff
                    current_timeout = self.timeout * (1 + 0.5 * retries)
                    logger.info(f"DeepSeek API request attempt {retries+1}/{self.max_retries} with timeout {current_timeout}s")

                    response = requests.post(endpoint, headers=headers, json=payload, timeout=current_timeout)
                    response_text = response.text

                    if response.status_code != 200:
                        error_msg = f"DeepSeek API HTTP error (status {response.status_code}): {response_text}"
                        logger.warning(error_msg)
                        last_error = requests.exceptions.HTTPError(error_msg, response=response)
                        if response.status_code >= 500:
                            retries += 1
                            if retries < self.max_retries:
                                wait_time = self.retry_delay * (2 ** retries)
                                logger.info(f"Server error, retrying in {wait_time}s...")
                                time.sleep(wait_time)
                                continue
                        raise last_error

                    try:
                        response_data = response.json()
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to decode JSON response: {e}\nResponse: {response_text}")
                        last_error = e
                        retries += 1
                        if retries < self.max_retries:
                            wait_time = self.retry_delay * (2 ** retries)
                            logger.info(f"JSON decode error, retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                        else:
                            raise last_error

                    if not response_data.get("choices"):
                        error_msg = f"No choices in response: {json.dumps(response_data, ensure_ascii=False)}"
                        logger.warning(error_msg)
                        last_error = ValueError(error_msg)
                        retries += 1
                        if retries < self.max_retries:
                            wait_time = self.retry_delay * (2 ** retries)
                            logger.info(f"Invalid response format, retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                        else:
                            raise last_error

                    message = response_data["choices"][0]["message"]["content"]

                    logger.info("DeepSeek API response received successfully")
                    logger.debug(f"DeepSeek API complete response: {json.dumps(response_data, ensure_ascii=False)}")
                    logger.debug(f"DeepSeek API message content: {message}")

                    # Update token usage and cost
                    if "usage" in response_data:
                        prompt_tokens = response_data["usage"].get("prompt_tokens", 0)
                        completion_tokens = response_data["usage"].get("completion_tokens", 0)
                        self.total_tokens += prompt_tokens + completion_tokens
                        self.total_cost += self._calculate_cost(prompt_tokens, completion_tokens)
                        logger.info(f"DeepSeek API token usage: {prompt_tokens + completion_tokens}, total cost: ${self.total_cost:.6f}")

                    # Create and return ChatResult
                    generation = ChatGeneration(message=AIMessage(content=message))
                    return ChatResult(generations=[generation])

                except (requests.exceptions.RequestException, ConnectionError, ValueError) as e:
                    last_error = e
                    logger.warning(f"Error during DeepSeek API request: {str(e)}")
                    retries += 1
                    self.failed_requests += 1

                    if retries < self.max_retries:
                        wait_time = self.retry_delay * (2 ** retries)
                        logger.info(f"Request error, retrying in {wait_time}s... (attempt {retries}/{self.max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed after {self.max_retries} attempts: {str(last_error)}")
                        error_message = f"Error calling DeepSeek API after {self.max_retries} attempts: {str(last_error)}"
                        generation = ChatGeneration(message=AIMessage(content=error_message))
                        return ChatResult(generations=[generation])

        except Exception as e:
            log_error(e, "DeepSeek API error")
            message = f"Error calling DeepSeek API: {str(e)}"
            generation = ChatGeneration(message=AIMessage(content=message))
            return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generate a response from the DeepSeek API."""
        try:
            # Convert LangChain messages to DeepSeek format with normalization
            deepseek_messages = self._normalize_messages(messages)

            # Prepare API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            is_reasoner = "r1" in self.model_name.lower() or "reason" in self.model_name.lower() or self._llm_type == "deepseek-reasoner"
            
            payload = {
                "model": self.model_name,
                "messages": deepseek_messages,
                "max_tokens": self.max_tokens,
            }
            if not is_reasoner:
                payload["temperature"] = self.temperature
                payload["top_p"] = self.top_p
                if stop:
                    payload["stop"] = stop

            # Log request details for debugging
            logger.debug(f"DeepSeek API request to {self.api_base}")
            logger.debug(f"Model: {self.model_name}")
            logger.debug(f"Payload: {json.dumps(payload, ensure_ascii=False)}")

            # Ensure API base URL is properly formatted and construct endpoint
            api_base = self.api_base.rstrip('/')
            endpoint = f"{api_base}/v1/chat/completions"

            # 实现重试机制
            retries = 0
            last_error = None

            while retries < self.max_retries:
                try:
                    # 使用指数退避策略计算当前超时时间
                    current_timeout = self.timeout * (1 + 0.5 * retries)  # 每次重试增加 50% 的超时时间
                    logger.info(f"DeepSeek API request attempt {retries+1}/{self.max_retries} with timeout {current_timeout}s")

                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            endpoint,
                            headers=headers,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=current_timeout)
                        ) as response:
                            response_text = await response.text()

                            # 检查响应状态
                            if response.status != 200:
                                error_msg = f"DeepSeek API HTTP error (status {response.status}): {response_text}"
                                logger.warning(error_msg)
                                last_error = aiohttp.ClientResponseError(
                                    request_info=response.request_info,
                                    history=response.history,
                                    status=response.status,
                                    message=error_msg,
                                    headers=response.headers
                                )
                                # 如果是服务器错误，重试
                                if response.status >= 500:
                                    retries += 1
                                    if retries < self.max_retries:
                                        wait_time = self.retry_delay * (2 ** retries)  # 指数退避
                                        logger.info(f"Server error, retrying in {wait_time}s...")
                                        await asyncio.sleep(wait_time)
                                        continue
                                # 如果是客户端错误，不重试
                                raise last_error

                            # 解析 JSON 响应
                            try:
                                response_data = json.loads(response_text)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to decode JSON response: {e}\nResponse: {response_text}")
                                last_error = e
                                retries += 1
                                if retries < self.max_retries:
                                    wait_time = self.retry_delay * (2 ** retries)
                                    logger.info(f"JSON decode error, retrying in {wait_time}s...")
                                    await asyncio.sleep(wait_time)
                                    continue
                                else:
                                    raise last_error

                            # 提取响应内容
                            if not response_data.get("choices"):
                                error_msg = f"No choices in response: {json.dumps(response_data, ensure_ascii=False)}"
                                logger.warning(error_msg)
                                last_error = ValueError(error_msg)
                                retries += 1
                                if retries < self.max_retries:
                                    wait_time = self.retry_delay * (2 ** retries)
                                    logger.info(f"Invalid response format, retrying in {wait_time}s...")
                                    await asyncio.sleep(wait_time)
                                    continue
                                else:
                                    raise last_error

                            message = response_data["choices"][0]["message"]["content"]

                            # 记录完整的响应内容用于调试
                            logger.info(f"DeepSeek API response received successfully")
                            logger.debug(f"DeepSeek API complete response: {json.dumps(response_data, ensure_ascii=False)}")
                            logger.debug(f"DeepSeek API message content: {message}")

                            # 更新令牌使用 and 成本
                            if "usage" in response_data:
                                prompt_tokens = response_data["usage"].get("prompt_tokens", 0)
                                completion_tokens = response_data["usage"].get("completion_tokens", 0)
                                self.total_tokens += prompt_tokens + completion_tokens
                                self.total_cost += self._calculate_cost(prompt_tokens, completion_tokens)
                                logger.info(f"DeepSeek API token usage: {prompt_tokens + completion_tokens}, total cost: ${self.total_cost:.6f}")

                            # 创建并返回 ChatResult
                            generation = ChatGeneration(message=AIMessage(content=message))
                            return ChatResult(generations=[generation])

                except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as e:
                    # 网络错误或超时错误，进行重试
                    last_error = e
                    logger.warning(f"Network error during DeepSeek API request: {str(e)}")
                    retries += 1
                    self.failed_requests += 1

                    if retries < self.max_retries:
                        wait_time = self.retry_delay * (2 ** retries)  # 指数退避
                        logger.info(f"Network error, retrying in {wait_time}s... (attempt {retries}/{self.max_retries})")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Failed after {self.max_retries} attempts: {str(last_error)}")
                        # 返回一个错误消息
                        error_message = f"Error calling DeepSeek API after {self.max_retries} attempts: {str(last_error)}"
                        generation = ChatGeneration(message=AIMessage(content=error_message))
                        return ChatResult(generations=[generation])

        except Exception as e:
            log_error(e, "DeepSeek API error")
            # Return a default message indicating the error
            message = f"Error calling DeepSeek API: {str(e)}"
            generation = ChatGeneration(message=AIMessage(content=message))
            return ChatResult(generations=[generation])


# Define a custom class for DeepSeek R1 model
class DeepSeekR1Model(DeepSeekChatModel):
    """DeepSeek R1 model wrapper for langchain"""

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "deepseek-reasoner"


from codedog.config.settings import settings


@lru_cache(maxsize=1)
def load_gpt_llm() -> BaseChatModel:
    """Load GPT 3.5 Model"""
    gpt35_model = settings.gpt35_model

    if settings.azure_openai:
        deployment_id = settings.azure_openai_deployment_id

        llm = AzureChatOpenAI(
            openai_api_type="azure",
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_api_base,
            api_version="2024-05-01-preview",
            azure_deployment=deployment_id,
            model=gpt35_model,
            temperature=0,
        )
    else:
        llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=gpt35_model,
            temperature=0,
        )
    return llm


@lru_cache(maxsize=1)
def load_gpt4_llm():
    """Load GPT 4 Model. Make sure your key have access to GPT 4 API. call this function won't check it."""
    gpt4_model = settings.gpt4_model

    if settings.azure_openai:
        deployment_id = settings.azure_openai_gpt4_deployment_id or settings.azure_openai_deployment_id

        llm = AzureChatOpenAI(
            openai_api_type="azure",
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_api_base,
            api_version="2024-05-01-preview",
            azure_deployment=deployment_id,
            model=gpt4_model,
            temperature=0,
        )
    else:
        llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=gpt4_model,
            temperature=0,
        )
    return llm


@lru_cache(maxsize=1)
def load_gpt4o_llm():
    """Load GPT-4o Model. Make sure your key have access to GPT-4o API."""
    gpt4o_model = settings.gpt4o_model

    if settings.azure_openai:
        deployment_id = settings.azure_openai_gpt4o_deployment_id or settings.azure_openai_deployment_id

        llm = AzureChatOpenAI(
            openai_api_type="azure",
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_api_base,
            api_version="2024-05-01-preview",
            azure_deployment=deployment_id,
            model=gpt4o_model,
            temperature=0,
        )
    else:
        llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=gpt4o_model,
            temperature=0,
        )
    return llm


@lru_cache(maxsize=1)
def load_deepseek_llm():
    """Load DeepSeek model"""
    llm = DeepSeekChatModel(
        api_key=settings.deepseek_api_key or "",
        model_name=settings.deepseek_model or "deepseek-chat",
        api_base=settings.deepseek_api_base or "https://api.deepseek.com",
        temperature=settings.deepseek_temperature,
        max_tokens=settings.deepseek_max_tokens,
        top_p=settings.deepseek_top_p,
        timeout=settings.deepseek_timeout,
        max_retries=settings.deepseek_max_retries,
        retry_delay=settings.deepseek_retry_delay,
    )
    return llm


@lru_cache(maxsize=1)
def load_deepseek_r1_llm():
    """Load DeepSeek R1 model"""
    llm = DeepSeekR1Model(
        api_key=settings.deepseek_api_key or "",
        model_name=settings.deepseek_r1_model or "deepseek-reasoner",
        api_base=settings.deepseek_r1_api_base or settings.deepseek_api_base or "https://api.deepseek.com",
        temperature=settings.deepseek_temperature,
        max_tokens=settings.deepseek_max_tokens,
        top_p=settings.deepseek_top_p,
        timeout=settings.deepseek_timeout,
        max_retries=settings.deepseek_max_retries,
        retry_delay=settings.deepseek_retry_delay,
    )
    return llm



def load_model_by_name(model_name: str) -> BaseChatModel:
    """Load a model by name

    Args:
        model_name: The name of the model to load. Can be:
            - "gpt-3.5" or any string starting with "gpt-3" for GPT-3.5 models
            - "gpt-4" or any string starting with "gpt-4" (except gpt-4o) for GPT-4 models
            - "gpt-4o" or "4o" for GPT-4o models
            - "deepseek" for DeepSeek models
            - "deepseek-r1" for DeepSeek R1 models
            - Any full OpenAI model name (e.g., "gpt-3.5-turbo-16k", "gpt-4-turbo", etc.)

    Returns:
        BaseChatModel: The loaded model

    Raises:
        ValueError: If the model name is not recognized
    """
    # Define standard model loaders
    model_loaders = {
        "gpt-3.5": load_gpt_llm,
        "gpt-4": load_gpt4_llm,
        "gpt-4o": load_gpt4o_llm,
        "4o": load_gpt4o_llm,
        "deepseek": load_deepseek_llm,
        "deepseek-r1": load_deepseek_r1_llm,
    }

    # Check for exact matches first
    if model_name in model_loaders:
        return model_loaders[model_name]()

    # Handle OpenAI model names with pattern matching
    if model_name.startswith("gpt-"):
        # Handle GPT-4o models
        if "4o" in model_name.lower():
            return load_gpt4o_llm()
        # Handle GPT-4 models
        elif model_name.startswith("gpt-4"):
            return load_gpt4_llm()
        # Handle GPT-3 models
        elif model_name.startswith("gpt-3"):
            return load_gpt_llm()
        # For any other GPT models, default to GPT-3.5
        else:
            logger.warning(f"Unrecognized GPT model name: {model_name}, defaulting to GPT-3.5")
            return load_gpt_llm()

    # If we get here, the model name is not recognized
    raise ValueError(f"Unknown model name: {model_name}. Available models: {list(model_loaders.keys())} or any OpenAI model name starting with 'gpt-'.")

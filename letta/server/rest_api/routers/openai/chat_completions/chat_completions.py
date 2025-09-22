import asyncio
from typing import TYPE_CHECKING, List, Optional, Union

from fastapi import APIRouter, Body, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from openai.types.chat.completion_create_params import CompletionCreateParams

from letta.agent import Agent
from letta.constants import DEFAULT_MESSAGE_TOOL, DEFAULT_MESSAGE_TOOL_KWARG, LETTA_MODEL_ENDPOINT
from letta.log import get_logger
from letta.schemas.message import Message, MessageCreate
from letta.schemas.user import User
from letta.server.rest_api.chat_completions_interface import ChatCompletionsStreamingInterface

# TODO this belongs in a controller!
from letta.server.rest_api.utils import get_letta_server, get_user_message_from_chat_completions_request, sse_async_generator

if TYPE_CHECKING:
    from letta.server.server import SyncServer

router = APIRouter(prefix="/v1", tags=["chat_completions"])

logger = get_logger(__name__)


@router.post(
    "/{agent_id}/chat/completions",
    response_model=None,
    operation_id="create_chat_completions",
    responses={
        200: {
            "description": "Successful response",
            "content": {"text/event-stream": {}},
        }
    },
)
async def create_chat_completions(
    agent_id: str,
    completion_request: CompletionCreateParams = Body(...),
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),
):
    """
    创建聊天完成请求的异步处理函数
    
    Args:
        agent_id: 代理ID，用于标识特定的代理
        completion_request: OpenAI聊天完成请求参数
        server: Letta服务器实例，通过依赖注入获取
        user_id: 用户ID，从请求头中获取
    
    Returns:
        StreamingResponse: 流式响应对象
    
    Raises:
        HTTPException: 当请求验证失败时抛出HTTP异常
    """
    # Validate and process fields
    # 验证请求必须是流式请求
    if not completion_request["stream"]:
        raise HTTPException(status_code=400, detail="Must be streaming request: `stream` was set to `False` in the request.")

    # 获取用户对象，如果用户ID不存在则使用默认用户
    actor = server.user_manager.get_user_or_default(user_id=user_id)

    # 加载指定的代理实例
    letta_agent = server.load_agent(agent_id=agent_id, actor=actor)
    # 获取代理的LLM配置
    llm_config = letta_agent.agent_state.llm_config
    # 验证模型端点类型必须是OpenAI且不能是Letta默认端点
    if llm_config.model_endpoint_type != "openai" or llm_config.model_endpoint == LETTA_MODEL_ENDPOINT:
        error_msg = f"You can only use models with type 'openai' for chat completions. This agent {agent_id} has llm_config: \n{llm_config.model_dump_json(indent=4)}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    # 检查请求中的模型是否与代理配置的模型一致
    model = completion_request.get("model")
    if model != llm_config.model:
        warning_msg = f"The requested model {model} is different from the model specified in this agent's ({agent_id}) llm_config: \n{llm_config.model_dump_json(indent=4)}"
        logger.warning(f"Defaulting to {llm_config.model}...")
        logger.warning(warning_msg)

    # 调用消息发送函数并返回流式响应
    return await send_message_to_agent_chat_completions(
        server=server,
        letta_agent=letta_agent,
        actor=actor,
        messages=get_user_message_from_chat_completions_request(completion_request),
    )


async def send_message_to_agent_chat_completions(
    server: "SyncServer",
    letta_agent: Agent,
    actor: User,
    messages: Union[List[Message], List[MessageCreate]],
    assistant_message_tool_name: str = DEFAULT_MESSAGE_TOOL,
    assistant_message_tool_kwarg: str = DEFAULT_MESSAGE_TOOL_KWARG,
) -> StreamingResponse:
    """
    向代理发送消息并返回聊天完成的流式响应
    
    Split off into a separate function so that it can be imported in the /chat/completion proxy.
    
    Args:
        server: Letta服务器实例
        letta_agent: 代理对象
        actor: 用户对象
        messages: 消息列表，可以是Message或MessageCreate类型
        assistant_message_tool_name: 助手消息工具名称，默认为DEFAULT_MESSAGE_TOOL
        assistant_message_tool_kwarg: 助手消息工具参数名称，默认为DEFAULT_MESSAGE_TOOL_KWARG
    
    Returns:
        StreamingResponse: 流式响应对象，包含SSE格式的数据流
    
    Raises:
        HTTPException: 当处理过程中发生错误时抛出HTTP异常
        ValueError: 当代理接口类型不正确时抛出值错误
    """
    # For streaming response
    try:
        # TODO: cleanup this logic
        # 获取代理的LLM配置
        llm_config = letta_agent.agent_state.llm_config

        # Create a new interface per request
        # 为每个请求创建新的聊天完成流式接口
        letta_agent.interface = ChatCompletionsStreamingInterface()
        streaming_interface = letta_agent.interface
        # 验证接口类型是否正确
        if not isinstance(streaming_interface, ChatCompletionsStreamingInterface):
            raise ValueError(f"Agent has wrong type of interface: {type(streaming_interface)}")

        # Allow AssistantMessage is desired by client
        # 设置助手消息的工具名称和参数，允许客户端自定义
        streaming_interface.assistant_message_tool_name = assistant_message_tool_name
        streaming_interface.assistant_message_tool_kwarg = assistant_message_tool_kwarg

        # Related to JSON buffer reader
        # 配置内部思考是否放在kwargs中，这与JSON缓冲区读取器相关
        streaming_interface.inner_thoughts_in_kwargs = (
            llm_config.put_inner_thoughts_in_kwargs if llm_config.put_inner_thoughts_in_kwargs is not None else False
        )

        # Offload the synchronous message_func to a separate thread
        # 启动流式接口
        streaming_interface.stream_start()
        # 将同步的消息处理函数卸载到单独的线程中执行，避免阻塞异步事件循环
        asyncio.create_task(
            asyncio.to_thread(
                server.send_messages,
                actor=actor,
                agent_id=letta_agent.agent_state.id,
                input_messages=messages,
                interface=streaming_interface,
                put_inner_thoughts_first=False,
            )
        )

        # return a stream
        # 返回流式响应，使用SSE（Server-Sent Events）格式
        return StreamingResponse(
            sse_async_generator(
                streaming_interface.get_generator(),
                usage_task=None,
                finish_message=True,
            ),
            media_type="text/event-stream",
        )

    except HTTPException:
        # 重新抛出HTTP异常，不做额外处理
        raise
    except Exception as e:
        # 捕获所有其他异常，打印错误信息和堆栈跟踪
        print(e)
        import traceback

        traceback.print_exc()
        # 将异常转换为HTTP 500错误并抛出
        raise HTTPException(status_code=500, detail=f"{e}")

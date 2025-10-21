import datetime
import glob
import os
import json
import traceback
import time

from memgpt.persistence_manager import LocalStateManager
from memgpt.config import AgentConfig
from .system import get_login_event, package_function_response, package_summarize_message, get_initial_boot_messages
from .memory import CoreMemory as Memory, summarize_messages
from .openai_tools import completions_with_backoff as create
from .utils import get_local_time, parse_json, united_diff, printd, count_tokens, get_schema_diff
from .constants import (
    FIRST_MESSAGE_ATTEMPTS,
    MAX_PAUSE_HEARTBEATS,
    MESSAGE_SUMMARY_WARNING_FRAC,
    MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC,
    MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST,
    CORE_MEMORY_HUMAN_CHAR_LIMIT,
    CORE_MEMORY_PERSONA_CHAR_LIMIT,
)
from .errors import LLMError
from .functions.functions import load_all_function_sets


def initialize_memory(ai_notes, human_notes):
    """
    初始化 Memory（核心记忆体）对象，并设定 AI（persona）与 human（人类）角色的笔记内容。

    参数:
        ai_notes (str): 需要用于初始化 persona（AI 角色）的笔记内容。
        human_notes (str): 需要用于初始化 human（人类角色）的笔记内容。

    返回:
        Memory: 已经设定好 persona 和 human 内容的核心记忆体对象。

    异常:
        ValueError: 如果 ai_notes 或 human_notes 为空则抛出该异常。
    """
    if ai_notes is None:
        raise ValueError(ai_notes)  # 检查 ai_notes 是否为 None，如果是则抛出异常
    if human_notes is None:
        raise ValueError(human_notes)  # 检查 human_notes 是否为 None，如果是则抛出异常
    memory = Memory(human_char_limit=CORE_MEMORY_HUMAN_CHAR_LIMIT, persona_char_limit=CORE_MEMORY_PERSONA_CHAR_LIMIT)  # 创建核心记忆体对象，并设置相应角色字符上限
    memory.edit_persona(ai_notes)  # 编辑/设定 persona（AI 角色）笔记内容
    memory.edit_human(human_notes)  # 编辑/设定 human（人类角色）笔记内容
    return memory  # 返回初始化完成的 Memory 对象


def construct_system_with_memory(system, memory, memory_edit_timestamp, archival_memory=None, recall_memory=None):
    full_system_message = "\n".join(
        [
            system,
            "\n",
            f"### Memory [last modified: {memory_edit_timestamp}]",
            f"{len(recall_memory) if recall_memory else 0} previous messages between you and the user are stored in recall memory (use functions to access them)",
            f"{len(archival_memory) if archival_memory else 0} total memories you created are stored in archival memory (use functions to access them)",
            "\nCore memory shown below (limited in size, additional information stored in archival / recall memory):",
            "<persona>",
            memory.persona,
            "</persona>",
            "<human>",
            memory.human,
            "</human>",
        ]
    )
    return full_system_message


def initialize_message_sequence(
    model,  # str: 使用的模型名称，例如 "gpt-4" 或 "gpt-3.5-turbo"
    system,  # str: 基础 system prompt，定义 AI 的行为准则
    memory,  # Memory: CoreMemory 实例，包含当前的 persona 和 human 内容
    archival_memory=None,  # Optional[List]: 存档记忆（Archival Memory），用户和 AI 之间过去的长期记忆列表
    recall_memory=None,  # Optional[List]: 召回记忆（Recall Memory），存储近期压缩或被截断的对话消息
    memory_edit_timestamp=None,  # Optional[str]: 上次 memory 被修改的时间戳，默认为当前本地时间
    include_initial_boot_message=True,  # bool: 是否在序列中插入初始开机消息，默认插入
):
    # 如果没有提供 memory_edit_timestamp，则使用当前本地时间
    if memory_edit_timestamp is None:
        memory_edit_timestamp = get_local_time()

    # 构建包含 memory 信息的完整 system 消息
    full_system_message = construct_system_with_memory(
        system, memory, memory_edit_timestamp, archival_memory=archival_memory, recall_memory=recall_memory
    )
    # 获取用户刚刚登录的事件（首条 user 消息）
    first_user_message = get_login_event()  # event letting MemGPT know the user just logged in

    # 判断是否需要包含初始的开机消息
    if include_initial_boot_message:
        # 针对 gpt-3.5 特殊处理启动消息，其余模型用默认启动消息
        if "gpt-3.5" in model:
            initial_boot_messages = get_initial_boot_messages("startup_with_send_message_gpt35")
        else:
            initial_boot_messages = get_initial_boot_messages("startup_with_send_message")
        # 组装完整的消息序列：system，initial boot messages, 和首条 user 消息
        messages = (
            [
                {"role": "system", "content": full_system_message},
            ]
            + initial_boot_messages
            + [
                {"role": "user", "content": first_user_message},
            ]
        )

    else:
        # 不包含初始 boot 消息时，仅包含 system 和 user 消息
        messages = [
            {"role": "system", "content": full_system_message},
            {"role": "user", "content": first_user_message},
        ]

    return messages


def get_ai_reply(
    model,                # str: 模型名称 (例如 "gpt-4", "gpt-3.5-turbo")
    message_sequence,     # List[dict]: 对话消息序列，每条消息为 dict 格式
    functions,            # List[dict]: 可用函数描述列表，用于函数调用增强
    function_call="auto", # str: 函数调用模式 (默认为 "auto"，也可指定具体函数名)
    context_window=None,  # Optional[int]: 上下文窗口大小限制，默认为 None，使用模型默认值
):
    try:
        # 调用 LLM 接口，传入模型信息、对话历史、函数信息等参数
        response = create(
            model=model,
            context_window=context_window,
            messages=message_sequence,
            functions=functions,
            function_call=function_call,
        )

        # special case for 'length'
        # 如果 finish_reason 是 "length"，表示达到最大上下文限制，抛出异常
        if response.choices[0].finish_reason == "length":
            raise Exception("Finish reason was length (maximum context length)")

        # catches for soft errors
        # 如果 finish_reason 不是 "stop" 或 "function_call"，说明不是正常结果，抛出异常
        if response.choices[0].finish_reason not in ["stop", "function_call"]:
            raise Exception(f"API call finish with bad finish reason: {response}")

        # unpack with response.choices[0].message.content
        # 返回完整 response，由调用方进一步解析
        return response

    except Exception as e:
        # 捕捉并向上传递异常
        raise e


class Agent(object):
    def __init__(
        self,
        config,
        model,
        system,
        functions,  # list of [{'schema': 'x', 'python_function': function_pointer}, ...]
        interface,
        persistence_manager,
        persona_notes,
        human_notes,
        messages_total=None,
        persistence_manager_init=True,
        first_message_verify_mono=True,
    ):
        # 代理配置参数对象
        self.config = config
        # LLM模型名称（如 gpt-4, gpt-3.5-turbo）
        self.model = model
        # 保存系统消息模板（用于重建记忆等，内容为system prompt）
        self.system = system

        # 可用函数映射: 从函数名到函数相关信息（schema和指针）
        # functions: dict, 每项为 function_name -> { "json_schema": ..., "python_function": ... }
        # 仅提取json_schema，供模型调用
        functions_schema = [f_dict["json_schema"] for f_name, f_dict in functions.items()]
        self.functions = functions_schema  # 模型可见schema列表
        # 保存python函数指针的映射，用于实际函数调用
        self.functions_python = {f_name: f_dict["python_function"] for f_name, f_dict in functions.items()}

        # 初始化记忆对象（如长短期记忆等），输入为人设笔记和用户笔记
        self.memory = initialize_memory(persona_notes, human_notes)
        # 用初始化的记忆对象，生成系统消息和初始消息序列
        self._messages = initialize_message_sequence(
            self.model,
            self.system,
            self.memory,
        )
        # 跟踪所有消息的总条数(不包含system消息，减一是去掉system类型)
        self.messages_total = messages_total if messages_total is not None else (len(self._messages) - 1)  # (-system)
        # 记录初始的消息数（通常用于判断“第一条消息”）
        # self.messages_total_init = self.messages_total  # 旧写法，暂时不用
        self.messages_total_init = len(self._messages) - 1
        printd(f"Agent initialized, self.messages_total={self.messages_total}")

        # 接口对象，负责显示/交互/输出，实现以下方法：
        # - internal_monologue: 处理内部思维
        # - assistant_message: 处理助手对话内容
        # - function_message: 处理函数相关内容
        # 不同interface类型可以自定义行为，如CLI输出、Discord消息等
        self.interface = interface

        # 持久化管理器，负责消息和状态的存取，实现接口方法：
        # - set_messages: 设置消息列表
        # - get_messages: 获取消息列表
        # - append_to_messages: 追加单条/多条消息
        self.persistence_manager = persistence_manager
        if persistence_manager_init:
            # 如需持久化，创建数据库/持久化中的新代理对象
            self.persistence_manager.init(self)

        # 用于心跳控制的“暂停心跳”起始时间
        self.pause_heartbeats_start = None
        # “暂停心跳”持续分钟数
        self.pause_heartbeats_minutes = 0

        # 首条消息时是否校验其内部思维的布尔标志位
        self.first_message_verify_mono = first_message_verify_mono

        # 对话内存压力预警的开关标志
        # 发送警告消息时置为True，避免过度重复警告
        # 进行摘要时自动置回False，恢复预警触发
        self.agent_alerted_about_memory_pressure = False

    @property
    def messages(self):
        """
        获取消息列表（对外部只读）。
        该属性用于访问代理的消息序列，外部仅允许读取，不能被直接修改。
        """
        return self._messages

    @messages.setter
    def messages(self, value):
        """
        禁止直接修改消息列表。
        如需修改消息，应通过专用方法管理消息队列的更改。
        """
        raise Exception("Modifying message list directly not allowed")

    def trim_messages(self, num):
        """Trim messages from the front, not including the system message"""
        self.persistence_manager.trim_messages(num)

        new_messages = [self.messages[0]] + self.messages[num:]
        self._messages = new_messages

    def prepend_to_messages(self, added_messages):
        """Wrapper around self.messages.prepend to allow additional calls to a state/persistence manager"""
        self.persistence_manager.prepend_to_messages(added_messages)

        new_messages = [self.messages[0]] + added_messages + self.messages[1:]  # prepend (no system)
        self._messages = new_messages
        self.messages_total += len(added_messages)  # still should increment the message counter (summaries are additions too)

    def append_to_messages(self, added_messages):
        """Wrapper around self.messages.append to allow additional calls to a state/persistence manager"""
        # 调用持久化管理器的方法，记录新增消息
        self.persistence_manager.append_to_messages(added_messages)

        # strip extra metadata if it exists
        # 移除每条新增消息中可能存在的 'api_response' 和 'api_args' 元数据字段
        for msg in added_messages:
            msg.pop("api_response", None)
            msg.pop("api_args", None)
        new_messages = self.messages + added_messages  # append  # 合并现有消息和待追加消息，生成新的消息列表

        self._messages = new_messages  # 更新消息列表
        self.messages_total += len(added_messages)  # 增加消息总数统计

    def swap_system_message(self, new_system_message):
        assert new_system_message["role"] == "system", new_system_message
        assert self.messages[0]["role"] == "system", self.messages

        self.persistence_manager.swap_system_message(new_system_message)

        new_messages = [new_system_message] + self.messages[1:]  # swap index 0 (system)
        self._messages = new_messages

    def rebuild_memory(self):
        """Rebuilds the system message with the latest memory object"""
        curr_system_message = self.messages[0]  # this is the system + memory bank, not just the system prompt
        # 生成包含最新 memory 对象的新 system 消息（消息序列的第一个元素即 system 消息）
        new_system_message = initialize_message_sequence(
            self.model,
            self.system,
            self.memory,
            archival_memory=self.persistence_manager.archival_memory,
            recall_memory=self.persistence_manager.recall_memory,
        )[0]

        # 计算旧 system 消息和新 system 消息内容的差异（diff 结果用于调试）
        diff = united_diff(curr_system_message["content"], new_system_message["content"])
        printd(f"Rebuilding system with new memory...\nDiff:\n{diff}")

        # Store the memory change (if stateful)
        self.persistence_manager.update_memory(self.memory)  # 持久化存储 memory 变更（如需状态管理）

        # Swap the system message out
        self.swap_system_message(new_system_message)  # 将系统消息替换为新内容

    ### Local state management
    def to_dict(self):
        return {
            "model": self.model,
            "system": self.system,
            "functions": self.functions,
            "messages": self.messages,
            "messages_total": self.messages_total,
            "memory": self.memory.to_dict(),
        }

    def save_to_json_file(self, filename):
        with open(filename, "w") as file:
            json.dump(self.to_dict(), file)

    def save(self):
        """Save agent state locally"""

        timestamp = get_local_time().replace(" ", "_").replace(":", "_")
        agent_name = self.config.name  # TODO: fix

        # save agent state
        filename = f"{timestamp}.json"
        os.makedirs(self.config.save_state_dir(), exist_ok=True)
        self.save_to_json_file(os.path.join(self.config.save_state_dir(), filename))

        # save the persistence manager too
        filename = f"{timestamp}.persistence.pickle"
        os.makedirs(self.config.save_persistence_manager_dir(), exist_ok=True)
        self.persistence_manager.save(os.path.join(self.config.save_persistence_manager_dir(), filename))

    @classmethod
    def load_agent(cls, interface, agent_config: AgentConfig):
        """Load saved agent state"""
        # TODO: support loading from specific file
        agent_name = agent_config.name

        # load state
        directory = agent_config.save_state_dir()
        json_files = glob.glob(os.path.join(directory, "*.json"))  # This will list all .json files in the current directory.
        if not json_files:
            print(f"/load error: no .json checkpoint files found")
            raise ValueError(f"Cannot load {agent_name}: does not exist in {directory}")

        # Sort files based on modified timestamp, with the latest file being the first.
        filename = max(json_files, key=os.path.getmtime)
        state = json.load(open(filename, "r"))

        # load persistence manager
        filename = os.path.basename(filename).replace(".json", ".persistence.pickle")
        directory = agent_config.save_persistence_manager_dir()
        printd(f"Loading persistence manager from {os.path.join(directory, filename)}")
        persistence_manager = LocalStateManager.load(os.path.join(directory, filename), agent_config)

        # need to dynamically link the functions
        # the saved agent.functions will just have the schemas, but we need to
        # go through the functions library and pull the respective python functions

        # Available functions is a mapping from:
        # function_name -> {
        #   json_schema: schema
        #   python_function: function
        # }
        # agent.functions is a list of schemas (OpenAI kwarg functions style, see: https://platform.openai.com/docs/api-reference/chat/create)
        # [{'name': ..., 'description': ...}, {...}]
        available_functions = load_all_function_sets()
        linked_function_set = {}
        for f_schema in state["functions"]:
            # Attempt to find the function in the existing function library
            f_name = f_schema.get("name")
            if f_name is None:
                raise ValueError(f"While loading agent.state.functions encountered a bad function schema object with no name:\n{f_schema}")
            linked_function = available_functions.get(f_name)
            if linked_function is None:
                raise ValueError(
                    f"Function '{f_name}' was specified in agent.state.functions, but is not in function library:\n{available_functions.keys()}"
                )
            # Once we find a matching function, make sure the schema is identical
            if json.dumps(f_schema) != json.dumps(linked_function["json_schema"]):
                # error_message = (
                #     f"Found matching function '{f_name}' from agent.state.functions inside function library, but schemas are different."
                #     + f"\n>>>agent.state.functions\n{json.dumps(f_schema, indent=2)}"
                #     + f"\n>>>function library\n{json.dumps(linked_function['json_schema'], indent=2)}"
                # )
                schema_diff = get_schema_diff(f_schema, linked_function["json_schema"])
                error_message = (
                    f"Found matching function '{f_name}' from agent.state.functions inside function library, but schemas are different.\n"
                    + "".join(schema_diff)
                )

                # NOTE to handle old configs, instead of erroring here let's just warn
                # raise ValueError(error_message)
                print(error_message)
            linked_function_set[f_name] = linked_function

        messages = state["messages"]
        agent = cls(
            config=agent_config,
            model=state["model"],
            system=state["system"],
            # functions=state["functions"],
            functions=linked_function_set,
            interface=interface,
            persistence_manager=persistence_manager,
            persistence_manager_init=False,
            persona_notes=state["memory"]["persona"],
            human_notes=state["memory"]["human"],
            messages_total=state["messages_total"] if "messages_total" in state else len(messages) - 1,
        )
        agent._messages = messages
        agent.memory = initialize_memory(state["memory"]["persona"], state["memory"]["human"])
        return agent

    @classmethod
    def load(cls, state, interface, persistence_manager):
        model = state["model"]
        system = state["system"]
        functions = state["functions"]
        messages = state["messages"]
        try:
            messages_total = state["messages_total"]
        except KeyError:
            messages_total = len(messages) - 1
        # memory requires a nested load
        memory_dict = state["memory"]
        persona_notes = memory_dict["persona"]
        human_notes = memory_dict["human"]

        # Two-part load
        new_agent = cls(
            model=model,
            system=system,
            functions=functions,
            interface=interface,
            persistence_manager=persistence_manager,
            persistence_manager_init=False,
            persona_notes=persona_notes,
            human_notes=human_notes,
            messages_total=messages_total,
        )
        new_agent._messages = messages
        return new_agent

    def load_inplace(self, state):
        self.model = state["model"]
        self.system = state["system"]
        self.functions = state["functions"]
        # memory requires a nested load
        memory_dict = state["memory"]
        persona_notes = memory_dict["persona"]
        human_notes = memory_dict["human"]
        self.memory = initialize_memory(persona_notes, human_notes)
        # messages also
        self._messages = state["messages"]
        try:
            self.messages_total = state["messages_total"]
        except KeyError:
            self.messages_total = len(self.messages) - 1  # -system

    @classmethod
    def load_from_json(cls, json_state, interface, persistence_manager):
        state = json.loads(json_state)
        return cls.load(state, interface, persistence_manager)

    @classmethod
    def load_from_json_file(cls, json_file, interface, persistence_manager):
        with open(json_file, "r") as file:
            state = json.load(file)
        return cls.load(state, interface, persistence_manager)

    def load_from_json_file_inplace(self, json_file):
        # Load in-place
        # No interface arg needed, we can use the current one
        with open(json_file, "r") as file:
            state = json.load(file)
        self.load_inplace(state)

    def verify_first_message_correctness(self, response, require_send_message=True, require_monologue=False):
        """Can be used to enforce that the first message always uses send_message"""
        response_message = response.choices[0].message

        # First message should be a call to send_message with a non-empty content
        if require_send_message and not response_message.get("function_call"):
            printd(f"First message didn't include function call: {response_message}")
            return False

        function_name = response_message["function_call"]["name"]
        if require_send_message and function_name != "send_message" and function_name != "archival_memory_search":
            printd(f"First message function call wasn't send_message or archival_memory_search: {response_message}")
            return False

        if require_monologue and (
            not response_message.get("content") or response_message["content"] is None or response_message["content"] == ""
        ):
            printd(f"First message missing internal monologue: {response_message}")
            return False

        if response_message.get("content"):
            ### Extras
            monologue = response_message.get("content")

            def contains_special_characters(s):
                special_characters = '(){}[]"'
                return any(char in s for char in special_characters)

            if contains_special_characters(monologue):
                printd(f"First message internal monologue contained special characters: {response_message}")
                return False
            # if 'functions' in monologue or 'send_message' in monologue or 'inner thought' in monologue.lower():
            if "functions" in monologue or "send_message" in monologue:
                # Sometimes the syntax won't be correct and internal syntax will leak into message.context
                printd(f"First message internal monologue contained reserved words: {response_message}")
                return False

        return True

    def handle_ai_response(self, response_message):
        """Handles parsing and function execution"""
        messages = []  # append these to the history when done

        # Step 2: check if LLM wanted to call a function
        if response_message.get("function_call"):
            # 当 assistant 回复包含 function_call 时，content 是内部独白，而不是聊天内容
            self.interface.internal_monologue(response_message.content)
            messages.append(response_message)  # 将助手的回复（含function_call）加入历史消息

            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors

            # Failure case 1: function name is wrong
            # 获取请求调用的函数名
            function_name = response_message["function_call"]["name"]
            try:
                # 从 self.functions_python 映射获取要调用的 Python 函数
                function_to_call = self.functions_python[function_name]
            except KeyError as e:
                # 未找到函数名，组装失败响应并返回
                error_msg = f"No function named {function_name}"
                function_response = package_function_response(False, error_msg)
                messages.append(
                    {
                        "role": "function",
                        "name": function_name,
                        "content": function_response,
                    }
                )  # extend conversation with function response
                self.interface.function_message(f"Error: {error_msg}")
                return messages, None, True  # force a heartbeat to allow agent to handle error

            # Failure case 2: function name is OK, but function args are bad JSON
            try:
                # 解析函数参数字符串为 dict
                raw_function_args = response_message["function_call"]["arguments"]
                function_args = parse_json(raw_function_args)
            except Exception as e:
                # 参数不是合法 JSON，组装失败响应并返回
                error_msg = f"Error parsing JSON for function '{function_name}' arguments: {raw_function_args}"
                function_response = package_function_response(False, error_msg)
                messages.append(
                    {
                        "role": "function",
                        "name": function_name,
                        "content": function_response,
                    }
                )  # extend conversation with function response
                self.interface.function_message(f"Error: {error_msg}")
                return messages, None, True  # force a heartbeat to allow agent to handle error

            # (Still parsing function args)
            # Handle requests for immediate heartbeat
            # 检查是否有 request_heartbeat 字段专门控制心跳
            heartbeat_request = function_args.pop("request_heartbeat", None)
            if not (isinstance(heartbeat_request, bool) or heartbeat_request is None):
                printd(
                    f"Warning: 'request_heartbeat' arg parsed was not a bool or None, type={type(heartbeat_request)}, value={heartbeat_request}"
                )
                heartbeat_request = None

            # Failure case 3: function failed during execution
            self.interface.function_message(f"Running {function_name}({function_args})")
            try:
                # 动态传递 self 以支持 bound methods
                function_args["self"] = self  # need to attach self to arg since it's dynamically linked
                
                # 开始计时函数调用
                function_start_time = time.time()
                function_response_string = function_to_call(**function_args)
                function_end_time = time.time()
                
                # 计算并输出函数执行耗时
                function_duration_ms = (function_end_time - function_start_time) * 1000
                from memgpt.interface import system_message
                system_message(f"[函数调用] {function_name} 执行耗时: {function_duration_ms:.2f} 毫秒")
                
                function_response = package_function_response(True, function_response_string)
                function_failed = False
            except Exception as e:
                # 执行函数出错，组装失败响应，将异常信息打印并反馈
                error_msg = f"Error calling function {function_name} with args {function_args}: {str(e)}"
                error_msg_user = f"{error_msg}\n{traceback.format_exc()}"
                printd(error_msg_user)
                function_response = package_function_response(False, error_msg)
                messages.append(
                    {
                        "role": "function",
                        "name": function_name,
                        "content": function_response,
                    }
                )  # extend conversation with function response
                self.interface.function_message(f"Error: {error_msg}")
                return messages, None, True  # force a heartbeat to allow agent to handle error

            # If no failures happened along the way: ...
            # Step 4: send the info on the function call and function response to GPT
            # 函数调用与结果均无异常时，将调用成功的信息和响应内容发送给对话历史和界面
            self.interface.function_message(f"Success: {function_response_string}")
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response

        else:
            # Standard non-function reply
            # 如果没有函数调用，仅为常规助手回复，则处理内部独白与消息队列
            self.interface.internal_monologue(response_message.content)
            messages.append(response_message)  # extend conversation with assistant's reply
            heartbeat_request = None
            function_failed = None

        # 返回本轮采集到的消息，心跳请求标志，及函数调用失败标志
        return messages, heartbeat_request, function_failed

    def step(self, user_message, first_message=False, first_message_retry_limit=FIRST_MESSAGE_ATTEMPTS, skip_verify=False):
        """Top-level event message handler for the MemGPT agent"""

        try:
            # Step 0: add user message
            # 如果存在用户消息，则向界面显示/处理该消息，同时将其包装成 message dict 并与历史消息拼接
            if user_message is not None:
                self.interface.user_message(user_message)  # 显示/处理用户消息到界面
                packed_user_message = {"role": "user", "content": user_message}  # 打包为 message 格式
                input_message_sequence = self.messages + [packed_user_message]  # 拼接历史消息和本轮用户消息
            else:
                input_message_sequence = self.messages  # 仅用历史消息（用于心跳等场景）

            # 检查最后一条消息是否为user，否则打印警告
            if len(input_message_sequence) > 1 and input_message_sequence[-1]["role"] != "user":
                printd(f"WARNING: attempting to run ChatCompletion without user as the last message in the queue")

            # Step 1: send the conversation and available functions to GPT
            # 判断是否需要执行首次消息额外的 verifier 逻辑
            # 若 skip_verify 为 False 且为首次消息，则针对首条回复进行多轮校验，确保其符合规范
            if not skip_verify and (first_message or self.messages_total == self.messages_total_init):
                printd(f"This is the first message. Running extra verifier on AI response.")
                counter = 0
                while True:
                    # 调用 LLM API 获取回复
                    ai_reply_start_time = time.time()
                    response = get_ai_reply(
                        model=self.model,
                        message_sequence=input_message_sequence,
                        functions=self.functions,
                        context_window=self.config.context_window,
                    )
                    ai_reply_end_time = time.time()
                    ai_reply_duration_ms = (ai_reply_end_time - ai_reply_start_time) * 1000
                    from memgpt.interface import system_message
                    system_message(f"[模型回复] get_ai_reply (首次验证) 执行耗时: {ai_reply_duration_ms:.2f} 毫秒")
                    # 校验首条消息的内容
                    if self.verify_first_message_correctness(response, require_monologue=self.first_message_verify_mono):
                        break

                    counter += 1
                    if counter > first_message_retry_limit:
                        raise Exception(f"Hit first message retry limit ({first_message_retry_limit})")

            else:
                # 普通情况：直接获取模型回复，无需多轮校验
                ai_reply_start_time = time.time()
                response = get_ai_reply(
                    model=self.model,
                    message_sequence=input_message_sequence,
                    functions=self.functions,
                    context_window=self.config.context_window,
                )
                ai_reply_end_time = time.time()
                ai_reply_duration_ms = (ai_reply_end_time - ai_reply_start_time) * 1000
                from memgpt.interface import system_message
                system_message(f"[模型回复] get_ai_reply (普通模式) 执行耗时: {ai_reply_duration_ms:.2f} 毫秒")

            # Step 2: check if LLM wanted to call a function
            # (if yes) Step 3: call the function
            # (if yes) Step 4: send the info on the function call and function response to LLM
            # 提取模型回复消息
            response_message = response.choices[0].message  # 获取 LLM 返回的主消息内容
            response_message_copy = response_message.copy()  # 备份一份原始消息内容
            # 处理 AI 回复，解析函数调用或普通回答
            all_response_messages, heartbeat_request, function_failed = self.handle_ai_response(response_message)

            # Add the extra metadata to the assistant response
            # (e.g. enough metadata to enable recreating the API call)
            # 向 assistant 的回复消息中添加额外元数据，方便后续复现 API 调用结果
            assert "api_response" not in all_response_messages[0]
            all_response_messages[0]["api_response"] = response_message_copy  # 保存完整原始回复信息
            assert "api_args" not in all_response_messages[0]
            all_response_messages[0]["api_args"] = {
                "model": self.model,
                "messages": input_message_sequence,
                "functions": self.functions,
            }

            # Step 4: extend the message history
            # 拼接所有新消息（用户输入+AI回复/函数结果等）到总消息队列
            if user_message is not None:
                all_new_messages = [packed_user_message] + all_response_messages
            else:
                all_new_messages = all_response_messages

            # Check the memory pressure and potentially issue a memory pressure warning
            # 检查本次 token 消耗是否接近窗口，决定是否触发警告
            current_total_tokens = response["usage"]["total_tokens"]  # 当前响应总 token 数
            active_memory_warning = False  # 默认无警告
            # 如果本次消耗的 token 超过警告阈值则触发内存压力警告
            if current_total_tokens > MESSAGE_SUMMARY_WARNING_FRAC * self.config.context_window:
                printd(
                    f"WARNING: last response total_tokens ({current_total_tokens}) > {MESSAGE_SUMMARY_WARNING_FRAC * self.config.context_window}"
                )
                # Only deliver the alert if we haven't already (this period)
                if not self.agent_alerted_about_memory_pressure:
                    active_memory_warning = True
                    self.agent_alerted_about_memory_pressure = True  # it's up to the outer loop to handle this
            else:
                printd(f"last response total_tokens ({current_total_tokens}) < {MESSAGE_SUMMARY_WARNING_FRAC * self.config.context_window}")

            # 更新消息历史
            self.append_to_messages(all_new_messages)
            # 返回新消息、心跳请求标志、函数失败标志、token警告标志
            return all_new_messages, heartbeat_request, function_failed, active_memory_warning

        except Exception as e:
            # 捕获 step 整体执行异常
            printd(f"step() failed\nuser_message = {user_message}\nerror = {e}")

            # If we got a context alert, try trimming the messages length, then try again
            # 如果触发了上下文长度超限的异常，则进行消息摘要后重试一次
            if "maximum context length" in str(e):
                # A separate API call to run a summarizer
                self.summarize_messages_inplace()

                # Try step again
                return self.step(user_message, first_message=first_message)
            else:
                printd(f"step() failed with openai.InvalidRequestError, but didn't recognize the error message: '{str(e)}'")
                raise e

    def summarize_messages_inplace(self, cutoff=None, preserve_last_N_messages=True):
        assert self.messages[0]["role"] == "system", f"self.messages[0] should be system (instead got {self.messages[0]})"

        # Start at index 1 (past the system message),
        # and collect messages for summarization until we reach the desired truncation token fraction (eg 50%)
        # Do not allow truncation of the last N messages, since these are needed for in-context examples of function calling
        token_counts = [count_tokens(str(msg)) for msg in self.messages]
        message_buffer_token_count = sum(token_counts[1:])  # no system message
        desired_token_count_to_summarize = int(message_buffer_token_count * MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC)
        candidate_messages_to_summarize = self.messages[1:]
        token_counts = token_counts[1:]
        if preserve_last_N_messages:
            candidate_messages_to_summarize = candidate_messages_to_summarize[:-MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST]
            token_counts = token_counts[:-MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST]
        printd(f"MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC={MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC}")
        printd(f"MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST={MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST}")
        printd(f"token_counts={token_counts}")
        printd(f"message_buffer_token_count={message_buffer_token_count}")
        printd(f"desired_token_count_to_summarize={desired_token_count_to_summarize}")
        printd(f"len(candidate_messages_to_summarize)={len(candidate_messages_to_summarize)}")

        # If at this point there's nothing to summarize, throw an error
        if len(candidate_messages_to_summarize) == 0:
            raise LLMError(
                f"Summarize error: tried to run summarize, but couldn't find enough messages to compress [len={len(self.messages)}, preserve_N={MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST}]"
            )

        # Walk down the message buffer (front-to-back) until we hit the target token count
        tokens_so_far = 0
        cutoff = 0
        for i, msg in enumerate(candidate_messages_to_summarize):
            cutoff = i
            tokens_so_far += token_counts[i]
            if tokens_so_far > desired_token_count_to_summarize:
                break
        # Account for system message
        cutoff += 1

        # Try to make an assistant message come after the cutoff
        try:
            printd(f"Selected cutoff {cutoff} was a 'user', shifting one...")
            if self.messages[cutoff]["role"] == "user":
                new_cutoff = cutoff + 1
                if self.messages[new_cutoff]["role"] == "user":
                    printd(f"Shifted cutoff {new_cutoff} is still a 'user', ignoring...")
                cutoff = new_cutoff
        except IndexError:
            pass

        message_sequence_to_summarize = self.messages[1:cutoff]  # do NOT get rid of the system message
        printd(f"Attempting to summarize {len(message_sequence_to_summarize)} messages [1:{cutoff}] of {len(self.messages)}")

        summary = summarize_messages(
            model=self.model, context_window=self.config.context_window, message_sequence_to_summarize=message_sequence_to_summarize
        )
        printd(f"Got summary: {summary}")

        # Metadata that's useful for the agent to see
        all_time_message_count = self.messages_total
        remaining_message_count = len(self.messages[cutoff:])
        hidden_message_count = all_time_message_count - remaining_message_count
        summary_message_count = len(message_sequence_to_summarize)
        summary_message = package_summarize_message(summary, summary_message_count, hidden_message_count, all_time_message_count)
        printd(f"Packaged into message: {summary_message}")

        prior_len = len(self.messages)
        self.trim_messages(cutoff)
        packed_summary_message = {"role": "user", "content": summary_message}
        self.prepend_to_messages([packed_summary_message])

        # reset alert
        self.agent_alerted_about_memory_pressure = False

        printd(f"Ran summarizer, messages length {prior_len} -> {len(self.messages)}")

    def heartbeat_is_paused(self):
        """Check if there's a requested pause on timed heartbeats"""

        # Check if the pause has been initiated
        if self.pause_heartbeats_start is None:
            return False

        # Check if it's been more than pause_heartbeats_minutes since pause_heartbeats_start
        elapsed_time = datetime.datetime.now() - self.pause_heartbeats_start
        return elapsed_time.total_seconds() < self.pause_heartbeats_minutes * 60

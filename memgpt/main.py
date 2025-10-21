import shutil
import configparser
import uuid
import logging
import glob
import os
import sys
import pickle
import traceback
import json
import time

import questionary
import typer

from rich.console import Console
from prettytable import PrettyTable
from .interface import print_messages

console = Console()

import memgpt.interface  # for printing to terminal
import memgpt.agent as agent
import memgpt.system as system
import memgpt.utils as utils
import memgpt.presets.presets as presets
import memgpt.constants as constants
import memgpt.personas.personas as personas
import memgpt.humans.humans as humans
from memgpt.persistence_manager import (
    LocalStateManager,
    InMemoryStateManager,
    InMemoryStateManagerWithPreloadedArchivalMemory,
    InMemoryStateManagerWithFaiss,
)
from memgpt.cli.cli import run, attach, version
from memgpt.cli.cli_config import configure, list, add
from memgpt.cli.cli_load import app as load_app
from memgpt.config import Config, MemGPTConfig, AgentConfig
from memgpt.constants import MEMGPT_DIR
from memgpt.agent import Agent
from memgpt.openai_tools import (
    configure_azure_support,
    check_azure_embeddings,
    get_set_azure_env_vars,
)
from memgpt.connectors.storage import StorageConnector

# 创建 Typer 应用实例，禁用漂亮的异常显示
app = typer.Typer(pretty_exceptions_enable=False)
# 注册 run 命令，用于运行 MemGPT 代理
app.command(name="run")(run)
# 注册 version 命令，用于显示版本信息
app.command(name="version")(version)
# 注册 attach 命令，用于连接到现有的代理会话
app.command(name="attach")(attach)
# 注册 configure 命令，用于配置 MemGPT 设置
app.command(name="configure")(configure)
# 注册 list 命令，用于列出可用的资源
app.command(name="list")(list)
# 注册 add 命令，用于添加新的资源
app.command(name="add")(add)
# load data commands
# 添加 load 子应用，用于数据加载相关的命令
app.add_typer(load_app, name="load")


def clear_line(strip_ui=False):
    if strip_ui:
        return
    if os.name == "nt":  # for windows
        console.print("\033[A\033[K", end="")
    else:  # for linux
        sys.stdout.write("\033[2K\033[G")
        sys.stdout.flush()


def save(memgpt_agent, cfg):
    filename = utils.get_local_time().replace(" ", "_").replace(":", "_")
    filename = f"{filename}.json"
    directory = os.path.join(MEMGPT_DIR, "saved_state")
    filename = os.path.join(directory, filename)
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        memgpt_agent.save_to_json_file(filename)
        print(f"Saved checkpoint to: {filename}")
        cfg.agent_save_file = filename
    except Exception as e:
        print(f"Saving state to {filename} failed with: {e}")

    # save the persistence manager too
    filename = filename.replace(".json", ".persistence.pickle")
    try:
        memgpt_agent.persistence_manager.save(filename)
        print(f"Saved persistence manager to: {filename}")
        cfg.persistence_manager_save_file = filename
    except Exception as e:
        print(f"Saving persistence manager to {filename} failed with: {e}")
    cfg.write_config()


def load(memgpt_agent, filename):
    if filename is not None:
        if filename[-5:] != ".json":
            filename += ".json"
        try:
            memgpt_agent.load_from_json_file_inplace(filename)
            print(f"Loaded checkpoint {filename}")
        except Exception as e:
            print(f"Loading {filename} failed with: {e}")
    else:
        # Load the latest file
        save_path = os.path.join(constants.MEMGPT_DIR, "saved_state")
        print(f"/load warning: no checkpoint specified, loading most recent checkpoint from {save_path} instead")
        json_files = glob.glob(os.path.join(save_path, "*.json"))  # This will list all .json files in the current directory.

        # Check if there are any json files.
        if not json_files:
            print(f"/load error: no .json checkpoint files found")
            return
        else:
            # Sort files based on modified timestamp, with the latest file being the first.
            filename = max(json_files, key=os.path.getmtime)
            try:
                memgpt_agent.load_from_json_file_inplace(filename)
                print(f"Loaded checkpoint {filename}")
            except Exception as e:
                print(f"Loading {filename} failed with: {e}")

    # need to load persistence manager too
    filename = filename.replace(".json", ".persistence.pickle")
    try:
        memgpt_agent.persistence_manager = InMemoryStateManager.load(
            filename
        )  # TODO(fixme):for different types of persistence managers that require different load/save methods
        print(f"Loaded persistence manager from {filename}")
    except Exception as e:
        print(f"/load warning: loading persistence manager from {filename} failed with: {e}")


# 设置应用回调函数，当没有指定子命令时调用此函数作为默认命令
@app.callback(invoke_without_command=True)  # make default command
# @app.command("legacy-run")
def legacy_run(
    ctx: typer.Context,
    persona: str = typer.Option(None, help="Specify persona"),
    human: str = typer.Option(None, help="Specify human"),
    model: str = typer.Option(constants.DEFAULT_MEMGPT_MODEL, help="Specify the LLM model"),
    first: bool = typer.Option(False, "--first", help="Use --first to send the first message in the sequence"),
    strip_ui: bool = typer.Option(False, "--strip_ui", help="Remove all the bells and whistles in CLI output (helpful for testing)"),
    debug: bool = typer.Option(False, "--debug", help="Use --debug to enable debugging output"),
    no_verify: bool = typer.Option(False, "--no_verify", help="Bypass message verification"),
    archival_storage_faiss_path: str = typer.Option(
        "",
        "--archival_storage_faiss_path",
        help="Specify archival storage with FAISS index to load (a folder with a .index and .json describing documents to be loaded)",
    ),
    archival_storage_files: str = typer.Option(
        "",
        "--archival_storage_files",
        help="Specify files to pre-load into archival memory (glob pattern)",
    ),
    archival_storage_files_compute_embeddings: str = typer.Option(
        "",
        "--archival_storage_files_compute_embeddings",
        help="Specify files to pre-load into archival memory (glob pattern), and compute embeddings over them",
    ),
    archival_storage_sqldb: str = typer.Option(
        "",
        "--archival_storage_sqldb",
        help="Specify SQL database to pre-load into archival memory",
    ),
    use_azure_openai: bool = typer.Option(
        False,
        "--use_azure_openai",
        help="Use Azure OpenAI (requires additional environment variables)",
    ),  # TODO: just pass in?
):
    if ctx.invoked_subcommand is not None:
        return

    typer.secho(
        "Warning: Running legacy run command. You may need to `pip install pymemgpt[legacy] -U`. Run `memgpt run` instead.",
        fg=typer.colors.RED,
        bold=True,
    )
    if not questionary.confirm("Continue with legacy CLI?", default=False).ask():
        return

    main(
        persona,
        human,
        model,
        first,
        debug,
        no_verify,
        archival_storage_faiss_path,
        archival_storage_files,
        archival_storage_files_compute_embeddings,
        archival_storage_sqldb,
        use_azure_openai,
        strip_ui,
    )


def main(
    persona,                                   # 指定 MemGPT 的 persona（人格/身份设定），控制 agent 说话和思维方式
    human,                                     # 指定与之交互的 "human"（用户/人类设定），影响对话上下文
    model,                                     # 使用的 LLM 模型名，如 "gpt-3.5-turbo"
    first,                                     # 是否由用户先发起对话（布尔值），决定主循环中谁是第一个提问者
    debug,                                     # 是否开启调试模式，True 会打印更多调试信息
    no_verify,                                 # 是否跳过某些验证检查（布尔值），一般用作开发测试
    archival_storage_faiss_path,               # 指定 FAISS 索引路径，从本地文件夹加载归档记忆（.index 及 .json 文档元数据所在目录）
    archival_storage_files,                    # 指定要预加载进归档记忆的文件集合（glob 文件匹配），直接加载内容
    archival_storage_files_compute_embeddings, # 指定要预加载并计算 embedding 的文件集合（glob 文件匹配），加载时会计算嵌入向量
    archival_storage_sqldb,                    # 指定要预加载进归档记忆的 SQL 数据库路径
    use_azure_openai,                         # 是否使用 Azure OpenAI 接口（布尔值），开启需要额外环境变量支持
    strip_ui,                                  # 是否精简 UI 输出（布尔值），主要用于简化终端显示
):
    # 设置界面显示参数
    memgpt.interface.STRIP_UI = strip_ui
    # 设置调试模式
    utils.DEBUG = debug
    # 默认日志级别为 CRITICAL
    logging.getLogger().setLevel(logging.CRITICAL)
    # 如果开启调试模式，则设置为 DEBUG
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 处理 Azure OpenAI 支持
    if use_azure_openai:
        configure_azure_support()
        check_azure_embeddings()
    else:
        # 检查是否错误配置了 Azure 环境变量
        azure_vars = get_set_azure_env_vars()
        if len(azure_vars) > 0:
            print(f"Error: Environment variables {', '.join([x[0] for x in azure_vars])} should not be set if --use_azure_openai is False")
            return

    # 检查是否传递了任何遗留参数（legacy flags），如果有则采用 legacy 初始化逻辑
    if any(
        (
            persona,
            human,
            model != constants.DEFAULT_MEMGPT_MODEL,
            archival_storage_faiss_path,
            archival_storage_files,
            archival_storage_files_compute_embeddings,
            archival_storage_sqldb,
        )
    ):
        memgpt.interface.important_message("⚙️ Using legacy command line arguments.")
        model = model
        if model is None:
            model = constants.DEFAULT_MEMGPT_MODEL
        memgpt_persona = persona
        # 选择 persona，如果没指定，则根据模型类型选择默认 persona
        if memgpt_persona is None:
            memgpt_persona = (
                personas.GPT35_DEFAULT if "gpt-3.5" in model else personas.DEFAULT,
                None,  # 代表 pymemgpt 包中的 personas 目录
            )
        else:
            try:
                # 尝试在自定义 personas 目录中查找
                personas.get_persona_text(memgpt_persona, Config.custom_personas_dir)
                memgpt_persona = (memgpt_persona, Config.custom_personas_dir)
            except FileNotFoundError:
                # 尝试在默认 personas 目录中查找
                personas.get_persona_text(memgpt_persona)
                memgpt_persona = (memgpt_persona, None)

        human_persona = human
        # 选择 human，如果没指定，则选用默认 human
        if human_persona is None:
            human_persona = (humans.DEFAULT, None)
        else:
            try:
                # 尝试在自定义 humans 目录中查找
                humans.get_human_text(human_persona, Config.custom_humans_dir)
                human_persona = (human_persona, Config.custom_humans_dir)
            except FileNotFoundError:
                # 尝试在默认 humans 目录中查找
                humans.get_human_text(human_persona)
                human_persona = (human_persona, None)

        print(persona, model, memgpt_persona)
        # 文件夹文件作为归档存储，且不重新计算 embedding
        if archival_storage_files:
            cfg = Config.legacy_flags_init(
                model,
                memgpt_persona,
                human_persona,
                load_type="folder",
                archival_storage_files=archival_storage_files,
                compute_embeddings=False,
            )
        # 指定 FAISS 索引作为归档存储，有 index + json 文件，重新计算 embedding
        elif archival_storage_faiss_path:
            cfg = Config.legacy_flags_init(
                model,
                memgpt_persona,
                human_persona,
                load_type="folder",
                archival_storage_files=archival_storage_faiss_path,
                archival_storage_index=archival_storage_faiss_path,
                compute_embeddings=True,
            )
        # 指定文件，并计算 embeddings
        elif archival_storage_files_compute_embeddings:
            print(model)
            print(memgpt_persona)
            print(human_persona)
            cfg = Config.legacy_flags_init(
                model,
                memgpt_persona,
                human_persona,
                load_type="folder",
                archival_storage_files=archival_storage_files_compute_embeddings,
                compute_embeddings=True,
            )
        # 指定 SQL 数据库
        elif archival_storage_sqldb:
            cfg = Config.legacy_flags_init(
                model,
                memgpt_persona,
                human_persona,
                load_type="sql",
                archival_storage_files=archival_storage_sqldb,
                compute_embeddings=False,
            )
        # 其它情况，采用默认逻辑
        else:
            cfg = Config.legacy_flags_init(
                model,
                memgpt_persona,
                human_persona,
            )
    else:
        # 没有 legacy flags，使用配置文件初始化
        cfg = Config.config_init()

    # 打印运行提示
    memgpt.interface.important_message("Running... [exit by typing '/exit', list available commands with '/help']")
    # 提醒用户当前模型如果是非默认模型可能不稳定
    if cfg.model != constants.DEFAULT_MEMGPT_MODEL:
        memgpt.interface.warning_message(
            f"⛔️ Warning - you are running MemGPT with {cfg.model}, which is not officially supported (yet). Expect bugs!"
        )

    # 持久化管理器设定
    if cfg.index:
        persistence_manager = InMemoryStateManagerWithFaiss(cfg.index, cfg.archival_database)
    elif cfg.archival_storage_files:
        print(f"Preloaded {len(cfg.archival_database)} chunks into archival memory.")
        persistence_manager = InMemoryStateManagerWithPreloadedArchivalMemory(cfg.archival_database)
    else:
        persistence_manager = InMemoryStateManager()

    # legacy 模式下如果计算了嵌入，给用户提示下次这样避免重复计算
    if archival_storage_files_compute_embeddings:
        memgpt.interface.important_message(
            f"(legacy) To avoid computing embeddings next time, replace --archival_storage_files_compute_embeddings={archival_storage_files_compute_embeddings} with\n\t --archival_storage_faiss_path={cfg.archival_storage_index} (if your files haven't changed)."
        )

    # 选择使用的 human 与 persona
    # Moved defaults out of FLAGS so that we can dynamically select the default persona based on model
    chosen_human = cfg.human_persona
    chosen_persona = cfg.memgpt_persona

    # 创建 MemGPT Agent 实例
    memgpt_agent = presets.use_preset(
        presets.DEFAULT_PRESET,
        None,  # no agent config to provide
        cfg.model,
        personas.get_persona_text(*chosen_persona),
        humans.get_human_text(*chosen_human),
        memgpt.interface,
        persistence_manager,
    )
    # 打印消息历史
    print_messages = memgpt.interface.print_messages
    print_messages(memgpt_agent.messages)

    # 如果加载类型为 sql，则将数据库中的内容加载进 agent 的记忆
    if cfg.load_type == "sql":  # TODO: move this into config.py in a clean manner
        if not os.path.exists(cfg.archival_storage_files):
            print(f"File {cfg.archival_storage_files} does not exist")
            return
        # Ingest data from file into archival storage
        else:
            print(f"Database found! Loading database into archival memory")
            data_list = utils.read_database_as_list(cfg.archival_storage_files)
            user_message = f"Your archival memory has been loaded with a SQL database called {data_list[0]}, which contains schema {data_list[1]}. Remember to refer to this first while answering any user questions!"
            for row in data_list:
                memgpt_agent.persistence_manager.archival_memory.insert(row)
            print(f"Database loaded into archival memory.")

    # 加载 agent 存档（如果存在）
    if cfg.agent_save_file:
        load_save_file = questionary.confirm(f"Load in saved agent '{cfg.agent_save_file}'?").ask()
        if load_save_file:
            load(memgpt_agent, cfg.agent_save_file)

    # 启动代理主循环
    run_agent_loop(memgpt_agent, first, no_verify, cfg, strip_ui, legacy=True)


def run_agent_loop(
    memgpt_agent,    # MemGPT Agent实例，处理主对话逻辑
    first,           # 是否由用户先开始对话（布尔值）
    no_verify=False, # 是否跳过首次消息验证检查，默认False，开发/调试用
    cfg=None,        # 配置信息对象（通常为命令行参数的解析结果或设置）
    strip_ui=False,  # 是否精简UI输出（True时简化界面提示，方便脚本化/无交互模式）
    legacy=False     # 指示是否运行在兼容旧版参数的"legacy"模式下
):
    # 初始化循环计数器和用户输入相关变量
    counter = 0
    user_input = None
    skip_next_user_input = False
    user_message = None
    USER_GOES_FIRST = first

    # 如果不是用户先开始，则等待用户按回车键开始对话
    if not USER_GOES_FIRST:
        console.input("[bold cyan]Hit enter to begin (will request first MemGPT message)[/bold cyan]")
        clear_line(strip_ui)
        print()

    # 初始化多行输入模式标志
    multiline_input = False
    while True:
        # 如果不跳过用户输入且满足条件（计数器大于0或用户先开始），则获取用户输入
        if not skip_next_user_input and (counter > 0 or USER_GOES_FIRST):
            # Ask for user input
            # 使用questionary获取用户输入，支持多行输入模式
            user_input = questionary.text(
                "Enter your message:",
                multiline=multiline_input,
                qmark=">",
            ).ask()
            clear_line(strip_ui)

            # Gracefully exit on Ctrl-C/D
            # 处理Ctrl-C/D退出情况
            if user_input is None:
                user_input = "/exit"

            # 去除输入末尾的空白字符
            user_input = user_input.rstrip()

            # 检查是否错误使用了!开头的命令
            if user_input.startswith("!"):
                print(f"Commands for CLI begin with '/' not '!'")
                continue

            # 不允许空输入
            if user_input == "":
                # no empty messages allowed
                print("Empty input received. Try again!")
                continue

            # Handle CLI commands
            # Commands to not get passed as input to MemGPT
            # 处理以/开头的CLI命令
            if user_input.startswith("/"):
                # 处理传统模式下的命令
                if legacy:
                    # legacy agent save functions (TODO: eventually remove)
                    # 处理加载命令
                    if user_input.lower() == "/load" or user_input.lower().startswith("/load "):
                        command = user_input.strip().split()
                        filename = command[1] if len(command) > 1 else None
                        load(memgpt_agent=memgpt_agent, filename=filename)
                        continue
                    # 处理退出命令，自动保存
                    elif user_input.lower() == "/exit":
                        # autosave
                        save(memgpt_agent=memgpt_agent, cfg=cfg)
                        break

                    # 处理保存聊天记录命令
                    elif user_input.lower() == "/savechat":
                        filename = utils.get_local_time().replace(" ", "_").replace(":", "_")
                        filename = f"{filename}.pkl"
                        directory = os.path.join(MEMGPT_DIR, "saved_chats")
                        try:
                            if not os.path.exists(directory):
                                os.makedirs(directory)
                            with open(os.path.join(directory, filename), "wb") as f:
                                pickle.dump(memgpt_agent.messages, f)
                                print(f"Saved messages to: {filename}")
                        except Exception as e:
                            print(f"Saving chat to {filename} failed with: {e}")
                        continue

                    # 处理保存命令
                    elif user_input.lower() == "/save":
                        save(memgpt_agent=memgpt_agent, cfg=cfg)
                        continue
                else:
                    # updated agent save functions
                    # 处理新版本的agent保存功能
                    if user_input.lower() == "/exit":
                        memgpt_agent.save()
                        break
                    elif user_input.lower() == "/save" or user_input.lower() == "/savechat":
                        memgpt_agent.save()
                        continue

                # 处理附加数据源命令
                if user_input.lower() == "/attach":
                    # 检查是否在传统模式下使用不支持的命令
                    if legacy:
                        typer.secho("Error: /attach is not supported in legacy mode.", fg=typer.colors.RED, bold=True)
                        continue

                    # TODO: check if agent already has it
                    # 获取可用的数据源选项并让用户选择
                    data_source_options = StorageConnector.list_loaded_data()
                    data_source = questionary.select("Select data source", choices=data_source_options).ask()

                    # attach new data
                    # 附加新数据源
                    attach(memgpt_agent.config.name, data_source)

                    # update agent config
                    # 更新agent配置
                    memgpt_agent.config.attach_data_source(data_source)

                    # reload agent with new data source
                    # TODO: maybe make this less ugly...
                    # 重新加载agent以使用新的数据源
                    memgpt_agent.persistence_manager.archival_memory.storage = StorageConnector.get_storage_connector(
                        agent_config=memgpt_agent.config
                    )
                    continue

                # 处理转储消息命令
                elif user_input.lower() == "/dump" or user_input.lower().startswith("/dump "):
                    # Check if there's an additional argument that's an integer
                    # 检查是否有额外的整数参数指定转储消息数量
                    command = user_input.strip().split()
                    amount = int(command[1]) if len(command) > 1 and command[1].isdigit() else 0
                    if amount == 0:
                        memgpt.interface.print_messages(memgpt_agent.messages, dump=True)
                    else:
                        memgpt.interface.print_messages(memgpt_agent.messages[-min(amount, len(memgpt_agent.messages)) :], dump=True)
                    continue

                # 处理转储原始消息命令
                elif user_input.lower() == "/dumpraw":
                    memgpt.interface.print_messages_raw(memgpt_agent.messages)
                    continue

                # 处理查看内存内容命令
                elif user_input.lower() == "/memory":
                    print(f"\nDumping memory contents:\n")
                    print(f"{str(memgpt_agent.memory)}")
                    print(f"{str(memgpt_agent.persistence_manager.archival_memory)}")
                    print(f"{str(memgpt_agent.persistence_manager.recall_memory)}")
                    continue

                # 处理切换模型命令
                elif user_input.lower() == "/model":
                    if memgpt_agent.model == "gpt-4":
                        memgpt_agent.model = "gpt-3.5-turbo-16k"
                    elif memgpt_agent.model == "gpt-3.5-turbo-16k":
                        memgpt_agent.model = "gpt-4"
                    print(f"Updated model to:\n{str(memgpt_agent.model)}")
                    continue

                # 处理弹出消息命令（撤销最近的消息）
                elif user_input.lower() == "/pop" or user_input.lower().startswith("/pop "):
                    # Check if there's an additional argument that's an integer
                    # 检查是否有额外的整数参数指定弹出消息数量
                    command = user_input.strip().split()
                    amount = int(command[1]) if len(command) > 1 and command[1].isdigit() else 3
                    print(f"Popping last {amount} messages from stack")
                    for _ in range(min(amount, len(memgpt_agent.messages))):
                        memgpt_agent.messages.pop()
                    continue

                # 处理重试命令
                elif user_input.lower() == "/retry":
                    # TODO this needs to also modify the persistence manager
                    print(f"Retrying for another answer")
                    # 回退到最后一条用户消息并重新发送
                    while len(memgpt_agent.messages) > 0:
                        if memgpt_agent.messages[-1].get("role") == "user":
                            # we want to pop up to the last user message and send it again
                            user_message = memgpt_agent.messages[-1].get("content")
                            memgpt_agent.messages.pop()
                            break
                        memgpt_agent.messages.pop()

                # 处理重新思考命令（修改最后一条助手消息的内容）
                elif user_input.lower() == "/rethink" or user_input.lower().startswith("/rethink "):
                    # TODO this needs to also modify the persistence manager
                    # 检查命令后是否有文本内容
                    if len(user_input) < len("/rethink "):
                        print("Missing text after the command")
                        continue
                    # 找到最后一条助手消息并修改其内容
                    for x in range(len(memgpt_agent.messages) - 1, 0, -1):
                        if memgpt_agent.messages[x].get("role") == "assistant":
                            text = user_input[len("/rethink ") :].strip()
                            memgpt_agent.messages[x].update({"content": text})
                            break
                    continue

                # 处理重写命令（修改最后一条助手消息的函数调用参数）
                elif user_input.lower() == "/rewrite" or user_input.lower().startswith("/rewrite "):
                    # TODO this needs to also modify the persistence manager
                    # 检查命令后是否有文本内容
                    if len(user_input) < len("/rewrite "):
                        print("Missing text after the command")
                        continue
                    # 找到最后一条助手消息并修改其函数调用的message参数
                    for x in range(len(memgpt_agent.messages) - 1, 0, -1):
                        if memgpt_agent.messages[x].get("role") == "assistant":
                            text = user_input[len("/rewrite ") :].strip()
                            args = json.loads(memgpt_agent.messages[x].get("function_call").get("arguments"))
                            args["message"] = text
                            memgpt_agent.messages[x].get("function_call").update({"arguments": json.dumps(args)})
                            break
                    continue

                # No skip options
                # 处理清除命令（重新创建agent）
                elif user_input.lower() == "/wipe":
                    memgpt_agent = agent.Agent(memgpt.interface)
                    user_message = None

                # 处理心跳命令
                elif user_input.lower() == "/heartbeat":
                    user_message = system.get_heartbeat()

                # 处理内存警告命令
                elif user_input.lower() == "/memorywarning":
                    user_message = system.get_token_limit_warning()

                # 处理切换多行输入模式命令
                elif user_input.lower() == "//":
                    multiline_input = not multiline_input
                    continue

                # 处理帮助命令
                elif user_input.lower() == "/" or user_input.lower() == "/help":
                    questionary.print("CLI commands", "bold")
                    for cmd, desc in USER_COMMANDS:
                        questionary.print(cmd, "bold")
                        questionary.print(f" {desc}")
                    continue

                # 处理未识别的命令
                else:
                    print(f"Unrecognized command: {user_input}")
                    continue

            else:
                # If message did not begin with command prefix, pass inputs to MemGPT
                # Handle user message and append to messages
                # 如果消息不是以命令前缀开始，则将输入传递给MemGPT
                # 处理用户消息并添加到消息列表中
                user_message = system.package_user_message(user_input)

        # 重置跳过下次用户输入的标志
        skip_next_user_input = False
        
        # 开始计时 - 从开始处理消息时开始
        loop_start_time = time.time()

        def process_agent_step(user_message, no_verify):
            # 调用 memgpt_agent 的 step 方法，处理 agent 的一步对话逻辑
            new_messages, heartbeat_request, function_failed, token_warning = memgpt_agent.step(
                user_message, first_message=False, skip_verify=no_verify
            )

            skip_next_user_input = False  # 是否跳过下一个用户输入的标志
            # 如果触发了 token 上限警告，则发送内存警告消息并跳过用户输入
            if token_warning:
                user_message = system.get_token_limit_warning()
                skip_next_user_input = True
            # 如果函数调用失败，则发送心跳消息并跳过用户输入
            elif function_failed:
                user_message = system.get_heartbeat(constants.FUNC_FAILED_HEARTBEAT_MESSAGE)
                skip_next_user_input = True
            # 如果收到心跳请求，则发送心跳消息并跳过用户输入
            elif heartbeat_request:
                user_message = system.get_heartbeat(constants.REQ_HEARTBEAT_MESSAGE)
                skip_next_user_input = True

            return new_messages, user_message, skip_next_user_input  # 返回新消息、更新后的用户消息以及是否跳过用户输入

        # 进入和处理 MemGPT agent 步骤的循环，直到成功或用户选择退出为止
        while True:
            try:
                # 如果启用了精简 UI，则不使用状态指示器
                if strip_ui:
                    # 调用 agent 的交互步骤
                    new_messages, user_message, skip_next_user_input = process_agent_step(user_message, no_verify)
                    break  # 成功处理后退出循环
                else:
                    # 显示“Thinking...”状态指示器，提示用户 agent 正在思考
                    with console.status("[bold cyan]Thinking...") as status:
                        # 调用 agent 的交互步骤
                        new_messages, user_message, skip_next_user_input = process_agent_step(user_message, no_verify)
                        break  # 成功处理后退出循环
            except Exception as e:
                # 捕获并打印 agent.step() 执行中的异常信息
                print("An exception ocurred when running agent.step(): ")
                traceback.print_exc()
                # 提示用户是否重试该步骤
                retry = questionary.confirm("Retry agent.step()?").ask()
                if not retry:
                    break  # 如果用户选择不重试，则退出循环

        # 结束计时并输出耗时统计
        if 'loop_start_time' in locals():
            loop_end_time = time.time()
            loop_duration_ms = (loop_end_time - loop_start_time) * 1000
            import memgpt.interface
            memgpt.interface.system_message(f"[循环 {counter + 1}] 处理耗时: {loop_duration_ms:.2f} 毫秒")
        
        counter += 1

    print("Finished.")


USER_COMMANDS = [
    ("//", "toggle multiline input mode"),
    ("/exit", "exit the CLI"),
    ("/save", "save a checkpoint of the current agent/conversation state"),
    ("/load", "load a saved checkpoint"),
    ("/dump <count>", "view the last <count> messages (all if <count> is omitted)"),
    ("/memory", "print the current contents of agent memory"),
    ("/pop <count>", "undo <count> messages in the conversation (default is 3)"),
    ("/retry", "pops the last answer and tries to get another one"),
    ("/rethink <text>", "changes the inner thoughts of the last agent message"),
    ("/rewrite <text>", "changes the reply of the last agent message"),
    ("/heartbeat", "send a heartbeat system message to the agent"),
    ("/memorywarning", "send a memory warning system message to the agent"),
    ("/attach", "attach data source to agent"),
]

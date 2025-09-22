#!/usr/bin/env python3
"""
Letta服务器启动脚本
支持直接运行: python main.py
也支持命令行参数，如: python main.py --port 8080 --debug
"""

import os
import sys
import typer

# 禁用composio版本检查输出
os.environ["COMPOSIO_DISABLE_VERSION_CHECK"] = "true"

def main():
    """启动Letta服务器的主函数"""
    # 检查是否有命令行参数
    if len(sys.argv) == 1:
        # 没有参数，直接启动服务器
        try:
            print("🚀 启动Letta服务器 (默认配置)...")
            from letta.server.rest_api.app import start_server
            start_server()
        except KeyboardInterrupt:
            print("\n⏹️  服务器已停止")
            sys.exit(0)
        except Exception as e:
            print(f"❌ 启动错误: {e}")
            sys.exit(1)
    else:
        # 有参数，显示提示信息但仍然支持基本启动
        if "--help" in sys.argv or "-h" in sys.argv:
            print("用法:")
            print("  python main.py                    # 使用默认设置启动服务器")
            print("  python main.py --server           # 通过CLI启动 (已弃用)")
            print("")
            print("推荐使用:")
            print("  python start_letta.py             # 简单启动")
            print("  python run_server.py --help       # 查看更多选项")
            print("  letta server                      # 官方命令")
            sys.exit(0)
        else:
            typer.secho(
                "注意: 直接使用 `python main.py` 已不推荐。请使用 `letta server` 或 `python start_letta.py`。",
                fg=typer.colors.YELLOW,
            )
            # 仍然尝试启动服务器
            try:
                from letta.server.rest_api.app import start_server
                start_server()
            except Exception as e:
                print(f"启动失败: {e}")
                sys.exit(1)

if __name__ == "__main__":
    main()

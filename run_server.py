#!/usr/bin/env python3
"""
独立的Letta服务器启动脚本
使用方法: python run_server.py [选项]

这个脚本允许您直接通过Python运行Letta服务器，而无需使用 'uv run letta server' 命令。
"""

import argparse
import os
import sys
from typing import Optional

# 确保项目路径在Python路径中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 禁用composio版本检查输出
os.environ["COMPOSIO_DISABLE_VERSION_CHECK"] = "true"


def main():
    """主函数：解析命令行参数并启动服务器"""
    parser = argparse.ArgumentParser(
        description="启动Letta服务器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_server.py                    # 使用默认设置启动服务器
  python run_server.py --port 8080        # 在端口8080启动服务器
  python run_server.py --host 0.0.0.0     # 允许外部访问
  python run_server.py --debug            # 启用调试模式
  python run_server.py --reload           # 启用热重载
        """
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=None,
        help="服务器运行端口 (默认: 8283)"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default=None,
        help="服务器运行主机地址 (默认: localhost)"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="启用调试模式"
    )
    
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="启用热重载 (开发模式)"
    )
    
    parser.add_argument(
        "--secure", 
        action="store_true",
        help="启用简单的安全访问控制"
    )
    
    parser.add_argument(
        "--localhttps", 
        action="store_true",
        help="设置本地HTTPS"
    )
    
    args = parser.parse_args()
    
    # 设置环境变量
    if args.localhttps:
        os.environ["LOCAL_HTTPS"] = "true"
    
    try:
        print("正在启动Letta服务器...")
        print(f"项目根目录: {project_root}")
        
        # 导入并启动服务器
        from letta.server.rest_api.app import start_server
        
        # 显示启动参数
        print(f"配置参数:")
        print(f"  - 主机: {args.host or 'localhost'}")
        print(f"  - 端口: {args.port or 8283}")
        print(f"  - 调试模式: {args.debug}")
        print(f"  - 热重载: {args.reload}")
        print(f"  - 安全模式: {args.secure}")
        print(f"  - HTTPS: {args.localhttps}")
        print()
        
        # 启动服务器
        start_server(
            port=args.port,
            host=args.host,
            debug=args.debug,
            reload=args.reload
        )
        
    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭服务器...")
        sys.exit(0)
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖项已正确安装。")
        print("您可以运行以下命令安装依赖:")
        print("  pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"启动服务器时发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

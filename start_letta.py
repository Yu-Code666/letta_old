#!/usr/bin/env python3
"""
简单的Letta服务器启动脚本
使用方法: python start_letta.py
"""

import os
import sys

# 确保项目路径在Python路径中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 禁用composio版本检查输出
os.environ["COMPOSIO_DISABLE_VERSION_CHECK"] = "true"

def check_dependencies():
    """检查必要的依赖项"""
    missing_deps = []
    
    try:
        import letta
    except ImportError:
        missing_deps.append("letta")
    
    try:
        import fastapi
    except ImportError:
        missing_deps.append("fastapi")
    
    try:
        import uvicorn
    except ImportError:
        missing_deps.append("uvicorn")
    
    if missing_deps:
        print(f"❌ 缺少以下依赖项: {', '.join(missing_deps)}")
        print("请运行以下命令安装依赖:")
        print("  pip install -e .")
        print("或者:")
        print("  pip install fastapi uvicorn")
        return False
    
    return True


if __name__ == "__main__":
    try:
        print("🚀 启动Letta服务器...")
        
        # 检查依赖项
        if not check_dependencies():
            sys.exit(1)
        
        # 导入并启动服务器
        from letta.server.rest_api.app import start_server
        
        # 使用默认配置启动服务器
        start_server()
        
    except KeyboardInterrupt:
        print("\n⏹️  服务器已停止")
        sys.exit(0)
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保在项目根目录下运行此脚本，并且所有依赖项已安装。")
        print("尝试运行: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动错误: {e}")
        print("详细错误信息:", str(e))
        sys.exit(1)

#!/usr/bin/env python3
"""
测试调用阿里云百炼平台 Kimi 2.5 模型（OpenAI 兼容接口）。

运行前安装依赖:
    pip install openai

设置 API Key 后运行:
    export DASHSCOPE_API_KEY=sk-your-key
    python test_kimi_bailian.py

或在当前 shell 一次执行:
    DASHSCOPE_API_KEY=sk-your-key python test_kimi_bailian.py
"""

import os
import sys


def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("错误: 请设置环境变量 DASHSCOPE_API_KEY", file=sys.stderr)
        print("示例: export DASHSCOPE_API_KEY=sk-your-key", file=sys.stderr)
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("错误: 请先安装 openai: pip install openai", file=sys.stderr)
        sys.exit(1)

    # 阿里云百炼灵积 OpenAI 兼容 endpoint，Kimi 2.5
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    model = "kimi-k2.5"
    print(f"正在调用模型: {model}")
    print("-" * 40)

    # 非流式调用
    print("\n[非流式] 请求: 用一句话介绍你自己")
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "用一句话介绍你自己"}],
    )
    reply = completion.choices[0].message.content
    print(f"回复: {reply}")

    # 流式调用
    print("\n[流式] 请求: 1+1等于几？只回答数字。")
    print("回复: ", end="", flush=True)
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "1+1等于几？只回答数字。"}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

    print("-" * 40)
    print("Kimi 2.5 调用测试完成。")


if __name__ == "__main__":
    main()

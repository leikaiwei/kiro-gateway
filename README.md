# Kiro Gateway (Fork)

本项目 fork 自 [jwadow/kiro-gateway](https://github.com/jwadow/kiro-gateway)，运行时代码跟随上游更新。完整功能和使用文档请参阅[上游 README](https://github.com/jwadow/kiro-gateway#readme)。

以下仅记录本 fork 相对于上游仍然存在的差异。

---

## Fork 差异

### CI / 镜像发布策略

- `.github/workflows/docker.yml` 拆分为测试、Docker 镜像验证与 release 发布三个阶段。
- Pull Request 和 `main` 分支推送仅运行验证，不推送镜像。
- 仅在 GitHub Release 发布时向 GHCR 推送 `linux/amd64` 与 `linux/arm64` 镜像。

## 已合并至上游的历史补丁

> 以下补丁已于 2026-04-18 通过上游 [PR #135](https://github.com/jwadow/kiro-gateway/pull/135) 合并。为便于追溯保留记录，但不再作为 fork 专属差异维护。

### ~~修复 Anthropic API 路径 token 估算严重低报~~

~~上游原有 fallback token 估算只计算 messages，忽略 tools 定义和 system prompt，导致 `input_tokens` 大幅偏低。~~

- ~~`kiro/tokenizer.py` - `count_message_tokens` 增加 `tool_use`/`tool_result` block 支持；增加 `count_system_tokens` 处理 Anthropic block list 格式；`estimate_request_tokens` 补全参数透传。~~
- ~~`kiro/streaming_anthropic.py` - 核心函数增加 `request_tools`/`request_system` 参数，使用完整请求进行 token 估算。~~
- ~~`kiro/routes_anthropic.py` - 路由层序列化 tools/system 并传入 streaming 函数。~~
- ~~`kiro/models_anthropic.py` - `AnthropicUsage` 增加 cache 字段透传支持。~~

### ~~修复 Anthropic 工具 token 低报与 cache usage 透传~~

- ~~`kiro/tokenizer.py` - `count_tools_tokens` 兼容 Anthropic flat 工具格式。~~
- ~~`kiro/streaming_anthropic.py` - `context_usage=0%` 时保留 fallback 估算；透传上游 `cache_read_input_tokens`/`cache_creation_input_tokens`。~~
- ~~`tests/unit/` - 增加回归测试覆盖 token 估算和 cache usage 场景。~~

# Kiro Gateway (Fork)

本项目 fork 自 [jwadow/kiro-gateway](https://github.com/jwadow/kiro-gateway)，完整文档请参阅原项目 README。

以下仅记录本 fork 相对于上游的补丁内容。

---

## 补丁列表

### 1. 修复 Anthropic API 路径 token 估算严重低报（PR #1）

上游的 fallback token 估算只计算 messages，完全忽略 tools 定义和 system prompt，导致 `input_tokens` 大幅偏低。

改动：
- `kiro/tokenizer.py` — `count_message_tokens` 新增 `tool_use`/`tool_result` block 支持；新增 `count_system_tokens` 处理 Anthropic block list 格式；`estimate_request_tokens` 补全参数透传
- `kiro/streaming_anthropic.py` — 核心函数新增 `request_tools`/`request_system` 参数，用 `estimate_request_tokens` 替代 `count_message_tokens`
- `kiro/routes_anthropic.py` — 路由层序列化 tools/system 并传入 streaming 函数
- `kiro/models_anthropic.py` — `AnthropicUsage` 新增 cache 字段 + `extra=allow`

### 2. 修复 Anthropic 工具 token 低报 + cache usage 透传（PR #2）

三个独立 bug 修复：

- `kiro/tokenizer.py` — `count_tools_tokens` 兼容 Anthropic flat 工具格式（原来只处理 OpenAI `type=function` 格式，flat 格式每个工具只算 4 token）
- `kiro/streaming_anthropic.py` — `context_usage=0%` 时不再用 0 覆盖 fallback 估算；新增 `_extract_cache_usage_fields()` 透传上游 `cache_read_input_tokens`/`cache_creation_input_tokens`
- `tests/unit/` — 新增 5 个回归测试覆盖以上修复

### 3. CI 流程优化

- `.github/workflows/docker.yml` — 拆分为 test / build-test-image / release-ghcr 三阶段，仅 release 事件推送 GHCR

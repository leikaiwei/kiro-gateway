# Kiro Gateway (Fork)

本项目 fork 自 [jwadow/kiro-gateway](https://github.com/jwadow/kiro-gateway)，运行时代码跟随上游更新。完整功能和使用文档请参阅[上游 README](https://github.com/jwadow/kiro-gateway#readme)。

以下仅记录本 fork 相对于上游仍然存在的差异。

---

## Fork 差异

### Endpoint 兼容性

上游将 API endpoint 从 `q.{region}.amazonaws.com` 迁移至 `runtime.{region}.kiro.dev`，但新 endpoint 对 SSO OIDC 账号返回 403。本 fork 还原了 `Content-Type` header，并支持通过环境变量覆盖 endpoint：

```yaml
environment:
  KIRO_API_HOST_TEMPLATE: "https://q.{region}.amazonaws.com"
  KIRO_Q_HOST_TEMPLATE: "https://q.{region}.amazonaws.com"
```

### profileArn 按账号是否具备来发，不按 auth 类型

请求 payload 与 `ListAvailableModels` 的 `profileArn` 改为**只要 `auth_manager.profile_arn` 有值就发**，不再限定 `auth_type == KIRO_DESKTOP`。

此前本 fork 曾反向修过一次（仅 Desktop 才发），当时把 403 归因为「SSO OIDC 发 profileArn 导致」。该结论不成立：走 SSO OIDC 的账号分两类，个人 Builder ID 本身没有 profileArn（取值为空，发不发都一样），而**企业 SSO 账号带 profileArn，上游要靠它识别订阅实体，不发才会 403**。按 auth 类型判断会把企业账号的 profileArn 一并丢掉。

判空即可覆盖两类账号，无需再分支：`converters_core.build_kiro_payload()` 本就对空值跳过该字段。

同时去掉了 `or PROFILE_ARN` 环境变量兜底 —— 该变量已在 `main.py` 的 `_add_env_overrides()` 里注入 `credentials.json`，再经 `KiroAuthManager` 归一到 `profile_arn`，在路由层重复兜底属冗余（且仅在 `credentials.json` 首次迁移时生效，多账号部署下本就不走这条路）。

### 首个 token 超时默认值

扩展思考模型（Opus 4.7 / 4.8）在产出首个 token 前可能思考 30–120 秒，上游默认的 `FIRST_TOKEN_TIMEOUT=15` 会过早触发重试，导致请求被反复取消。本 fork 将默认值提升至 120 秒。非思考模型如需更快重试，可通过环境变量下调：

```yaml
environment:
  FIRST_TOKEN_TIMEOUT: "30"
```

### Claude 5 系列支持原生 adaptive thinking

`NATIVE_THINKING_SUPPORTED_MODELS` 增加 `claude-opus-5` 与 `claude-sonnet-5`。列表按**子串**匹配归一化后的 model id，故 `claude-opus-5` 亦覆盖后续小版本（如 `claude-opus-5.1`）。

此前 Claude 5 不在列表内，只能走 `FAKE_REASONING`：往用户消息前置 `<thinking_mode>` 标签让模型把思考写进正文，再由 `ThinkingParser` 解析回 `thinking_delta`。这条路的思考 token 是**实打实生成的**，没有加速通道，且必须整篇吐完才轮到答案。

生效条件三者缺一不可：

1. `KIRO_NATIVE_THINKING_MODE` 设为 `auto` 或 `force`（默认 `off`，整套原生逻辑休眠）
2. 模型命中 `NATIVE_THINKING_SUPPORTED_MODELS`
3. `auto` 模式下客户端须发 `thinking={"type":"adaptive","effort":...}`；发 `budget_tokens` 会映射成 `effort=None` 而回落假思考

**Claude Code / LiteLLM 发的是 `budget_tokens`，因此实际流量只有 `force` 模式才切得到原生路径。**

2026-08-10 实测确认：原生字段 `thinking` / `output_config` 在**旧 endpoint `q.us-east-1.amazonaws.com` 上同样可用**，Kiro 侧无 `ValidationException`，无标签泄漏。故本改动不依赖切换 endpoint。

附带效果：走原生路径时 `inject_thinking_tags()` 不被调用，`Client requested thinking budget N exceeds cap` 警告随之消失。仅下调 `FAKE_REASONING_MAX_TOKENS` 只能消掉「客户端未指定 budget」那一条来路的警告，客户端自带 `budget_tokens` 或 `effort=max` 换算出的超额值仍会触发。

### fix: force 模式不再覆盖客户端显式关闭思考

`thinking={"type":"disabled"}` 映射成 `effort=None`，与「客户端未指定 effort」走同一分支，被 `force` 模式兜底成 `"high"` —— 显式关闭反而被打开。`build_native_thinking_config()` 增加 `client_disabled` 参数，opt-out 在任何模式下优先。

### fix: 假思考说明文字跟随本次请求状态

`get_thinking_system_prompt_addition()` 原本只按全局 `FAKE_REASONING_ENABLED` 判断，导致原生思考接管（或客户端关闭思考）时，system prompt 里仍插入「把推理包在 `<thinking>` 标签里」的说明 —— 模型会同时产出原生思考和假标签，而后者已无人解析。改为仅在本次请求真的注入假标签时才添加。

### fix(parsers): 上游计费事件从未被匹配，credit 一直丢失

已提上游 [PR #280](https://github.com/jwadow/kiro-gateway/pull/280)，合入后本节可删。

`EVENT_PATTERNS` 里的模式是 `{"usage":`，而上游真实的计费事件形如 `{"unit":"credit","unitPlural":"credits","usage":<float>}` —— `usage` 是第三个键，`buffer.find('{"usage":')` 永远撞不上。于是 `metering_data` 恒为 `None`，`streaming_openai.py` 里那句 `credits_used` 输出从未生效，Anthropic 路径也拿不到 credit（#135 加的 cache usage 透传同样卡在这条模式上，一直是死代码）。

`git log -S'"unit":' -- kiro/` 在全部提交里零命中 —— 不是有意丢弃，是照 AWS SDK 的字段名（`MeteringEvent { usage, unit }`）猜 wire 形态猜错了，`unit` 序列化时排在前面。

新增独立的 `metering` 事件类型，不复用 `usage`：前者是 float 的 credit，后者是 cache token 的 dict，复用会在上游哪天真发 `{"usage":{...}}` 时互相覆盖。模式只取 `{"unit":`，unit 值的校验放在 `_process_metering_event` 里 —— 写成 `{"unit":"credit"` 的话，上游 JSON 多一个空格就静默失效。

**credit 必须累加**：一个下游请求可能触发多次上游调用（工具调用），实测同一个 `/v1/messages` 的上游流里有两个计费事件 `0.0930 + 0.5290`，取最后一个少算 15%。两个 API、流式与非流式四条路都覆盖。

每个请求记一行 `[Credit] req=<id> model=<model> credits=<float> calls=<n>`，`req=` 是外部按日志对账时的去重键。

OpenAI 的 `usage.credits_used` 由 `EXPOSE_CREDITS_USED` 控制，**默认关**：生产流量全走 `/v1/messages`，OpenAI 端点零流量，不值得让 bug 修复顺带改变对外 payload 契约。Anthropic 路径只累加与记日志，payload 一个字节不改。

> **别把这个字段改名叫 `cost`。** credit 不是钱（加购价 $0.04/credit），而 LiteLLM 的 `Usage` 有一个显式的 `cost: float | None` 字段会被当成真实费用采纳，改名等于直接污染下游计费与 spend 记账。

副作用只有一处：OpenAI 路径截断检测的 `received_usage` 恢复生效（原本 `received_usage or received_context_usage` 是两条腿，一直只有 `context_usage` 单腿承重），判定会略微变宽松。Anthropic 路径压根不看 metering，不受影响。上线后需盯 `Content truncated by Kiro API` 日志：减少是预期的（少了误判），**归零则要查是否漏判真截断**。

### fix(anthropic): 网关自产的 WebSearch 内容块自己不收，触发一次会话即永久 422

上游同款问题，已在 [PR #259](https://github.com/jwadow/kiro-gateway/pull/259)（server tool 块）与 [PR #163](https://github.com/jwadow/kiro-gateway/pull/163)（document 块）待合入，本 fork 先行取用。

`ContentBlock` 这个 Union 只认六种块，而网关自己会往客户端吐 `server_tool_use` / `web_search_tool_result` / `web_search_result` —— Path A（`mcp_tools.py`）和 Path B（`streaming_anthropic.py` 流中途拦截 `web_search` 工具调用）**两条路都吐**。客户端按 Anthropic 协议把这些块留在消息历史里，下一轮原样回传，Pydantic 在进业务逻辑之前就 422。结果块一进历史就不会消失，**该会话此后每一轮都 422，直到用户新开会话**。`model_config = {"extra": "allow"}` 救不了，它只对顶层字段宽松，Union 成员匹配照样严格。

生产 8 天日志（181，16 容器）实测 150 条 422：121 条因 `server_tool_use`（102 条同时含结果块），29 条因 `document` 块（PDF / 文本附件）。被拒的 id 形如 `srvtoolu_<32hex>`，正是 `mcp_tools.py` 自己生成的格式。**产生方是 Path B 而非 Path A**：121 条里 74 条在 `server_tool_use` 之前有 `thinking` 块、47 条有模型自述 text，而 Path A 的 SSE 序列固定为 `server_tool_use → web_search_tool_result → summary text`，不可能带这种前缀。同期 Path A 被调用 460 次，抽出的 query 长度中位 68、最长 163 字符，全部是单轮纯搜索请求，工作正常。

三处改动：

1. **`models_anthropic.py`** —— Union 补 `ServerToolUseContentBlock`、`WebSearchToolResultContentBlock`（含 `web_search_tool_result_error` 分支）、`DocumentContentBlock`。校验并未整体放宽，缺 `text` 的畸形 text 块仍然被拒。
2. **`converters_anthropic.py`** —— `server_tool_use` 计入 tool_calls，`web_search_tool_result` 转成挂到紧随其后 user 消息上的 tool_result（复用 `generate_search_summary` 渲染）。走 tool_result 而不是拼进正文，是为了保住 `build_kiro_history()` 依赖的 tool_use/tool_result 配对；对话以 assistant 结尾时补一条 user 消息承接，避免结果被丢。document 块则在 `convert_anthropic_content_to_text()` 里摊平：`source.type == "text"` 原样内联，二进制来源（PDF 等）只留 `[Document: 名字 (media_type) — content not available to the model]` 占位。

   > **PDF 不做文本解析是有意取舍。** 上游 PR #163 手写了约 90 行正则 + zlib 的 PDF 抽取器，对 372KB 的真实文档既不可靠又会把二进制噪声灌进 prompt。占位符已让模型知道有附件而不再 422，需要内容时客户端本就会另发工具读取。

3. **`routes_anthropic.py`** —— Path A 的 early return 加守卫。只补模型是不够的：请求放行后依然会无条件 early return，而 `extract_query_from_messages()` 只看 `messages[0]`（其注释自陈 `LIMITATION: single-turn`），在多轮回放里会拿第一条用户消息当搜索词重搜一遍，模型永远轮不到回答 —— 422 变成答非所问，一样是坏的。`has_replayed_server_search()` 检测到历史里已有服务端搜索块就跳过 Path A，走正常推理。

   配套：跳过 Path A 时须一并剥掉那个 native `web_search` 工具。它没有 `input_schema`，转换时 `inputSchema.json` 被兜底成 `{}`，模型调用它带不出 `query`，Path B 拦截会因缺 query 静默空转 —— 搜索能力会无声失效。剥掉后由 Path B 的自动注入换成带 `query` schema 的可用定义，搜索由 Path B 承接，行为连贯。

**未采纳** PR #259 的 `streaming_anthropic.py` 部分（+319 行、5 轮上限的内部续写循环）：它解决的是上游 issue #258「Path B 搜完直接结束、模型来不及综述」的体验问题，不是 422，风险与收益不匹配。

Path A 守卫上游无人做过，值得单独回一个 PR。

### CI / 镜像发布策略

- `.github/workflows/docker.yml` 拆分为测试、Docker 镜像验证与 release 发布三个阶段。
- Pull Request 和 `main` 分支推送仅运行验证，不推送镜像。
- 仅在 GitHub Release 发布时向 GHCR 推送 `linux/amd64` 与 `linux/arm64` 镜像。

## 临时合并的上游 PR

> 以下 PR 尚未被上游 `main` 合入，为支持 Claude Opus 4.8 临时 cherry-pick 到本 fork。**待上游合入后应还原这些提交，改为同步上游 main。**

### fix(models): 接受 inline system role（[PR #195](https://github.com/jwadow/kiro-gateway/pull/195)）

Claude Code 使用 Opus 4.8 时会发送 inline `system` 消息，原有 `AnthropicMessage.role` 仅接受 `user`/`assistant`，导致 422 错误。扩展为接受 `system` 并由 `normalize_message_roles` 下游处理。

### fix(model_resolver): 解析 MODEL_ALIASES（[PR #184](https://github.com/jwadow/kiro-gateway/pull/184)）

`get_model_id_for_kiro()` 未先检查 aliases 就直接 normalize，导致自定义别名（如 Cursor IDE 用户需要的自定义模型名）无法正确路由到 Kiro API。

### feat(thinking): 支持原生 adaptive reasoning（[PR #192](https://github.com/jwadow/kiro-gateway/pull/192)）

添加 `KIRO_NATIVE_THINKING_MODE` 环境变量（off/auto/force），支持 Opus 4.8 的原生 adaptive thinking。解析 `reasoningContentEvent` 帧，通过 OpenAI/Anthropic 响应路径输出 reasoning 内容。

---

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

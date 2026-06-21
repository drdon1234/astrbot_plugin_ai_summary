# AI 引用内容总结插件架构文档

本文档描述 `astrbot_plugin_ai_summary` 的当前运行时架构。

## 1. 系统概览

`astrbot_plugin_ai_summary` 是一个 AstrBot 插件，用于从引用消息中提取文字、图片、视频和合并转发集合；远端媒体会下载到本地，视频会进行 ASR 转写和可选视觉兜底，最终和图片、引用文字、转发集合展开文本一起交给 LLM 生成总结。

它有三条主要入口：

1. 普通消息总结流程：基于总结命令触发，要求引用含视频、图片、文字、图文混合或合并转发的消息。
2. 总结问答流程：私聊和群聊都通过引用插件的总结或问答回复触发，引用正文直接作为问题；问答使用同一私聊或群聊内的总结记录和最近问答作为临时上下文。
3. 管理员连通性测试流程：基于可配置关键词触发，只允许私聊中的 `permissions.admin_id` 对应账号使用。

核心链路如下：

```mermaid
flowchart TD
    A["AstrBot 消息事件"] --> B["main.py 入口"]
    B --> C{"是否命中总结触发条件"}
    C -- "否" --> Z["忽略"]
    C -- "是" --> E{"是否有可总结引用内容"}
    E -- "否" --> Z
    E -- "是" --> F["准备文字/图片/视频/转发"]
    E -- "是" --> D["在有候选时提取可选 user_hint"]
    F --> G["AISummaryManager"]
    G --> H["汇总引用文字和视频 ASR"]
    G --> R["附带引用图片"]
    H --> I["引用文字/语音转写 transcript"]
    I --> J{"视频是否需要视觉兜底"}
    J -- "需要" --> K["视觉观察 visual"]
    J -- "不需要" --> L["visual 为空"]
    D --> M["LLM 总结"]
    I --> M
    R --> M
    K --> M
    L --> M
    M --> N["返回总结消息"]
    N --> O["保存到 cache/runtime/qa_records"]
    O --> P{"引用总结或问答回复"}
    P --> Q["基于绑定总结回答"]
```

## 2. 模块边界

```text
main.py
core/
  config.py
  output_render.py
  qa_runtime.py
  qa_store.py
  summary/
    __init__.py
    manager.py
    llm_client.py
    llm_provider_defs.py
    prompts.py
    asr_runtime.py
    asr_worker.py
```

### `main.py`

插件入口层，负责：

- 注册 AstrBot 插件
- 监听消息事件
- 判断权限和触发条件
- 提取引用消息里的文字、图片、视频源和合并转发 ID
- 通过 `get_forward_msg` 展开合并转发节点，并把节点文本按顺序合并为一个总结候选
- 下载远端视频和图片文件
- 清理下载产物
- 保存成功总结作为临时问答知识库
- 处理引用插件回复触发的总结问答
- 转发到 `AISummaryManager`

### `core/config.py`

配置解析层，负责：

- 把 AstrBot WebUI 配置转换成 `AISummaryConfig`
- 解析触发器、LLM、权限、输出控制和管理员调试配置
- 生成插件运行所需的默认缓存路径

### `core/output_render.py`

输出渲染层，负责：

- 使用 Pillow 将纯文本或常见 Markdown 结构绘制为温和浅色单列卡片 PNG
- 固定加载插件本地 `resource/font/NotoSansCJKsc-Regular.otf` 与 `resource/font/NotoSansCJKsc-Bold.otf` 字体文件，避免依赖系统中文字体
- 通过 `output.image_font_size` 控制图片正文基础字号，标题和小标题按比例缩放
- 不依赖外部截图渲染器

### `core/qa_runtime.py`

总结问答运行时辅助层，负责生成私聊 / 群聊隔离的知识库 scope，并提供缺失记录时的用户提示。

### `core/qa_store.py`

轻量知识库存储层，负责：

- 把首次总结结果保存为 JSON 记录
- 在每条总结记录上保存最近几轮问答 pair
- 按私聊用户或群聊隔离记录
- 在读取记录时刷新 `last_accessed_at`
- 按配置的空闲时间清理过期记录

### `core/summary/manager.py`

总结编排层，负责：

- 调度 ASR 准备和模型准备
- 从本地视频提取音频
- 执行本地 ASR 子进程
- 决定是否启用视觉兜底
- 组织 LLM 总结请求，可使用 AstrBot 已配置的 AI 或插件自定义接口
- 基于已保存的总结文本回答后续问题
- 提供管理员连通性测试能力

### `core/summary/llm_client.py`

插件自定义 LLM 协议适配层，负责：

- 将通用聊天 payload 转成不同厂商的 HTTP 请求
- 支持 OpenAI、Azure OpenAI、Anthropic、Gemini、Ollama 等协议
- 从响应中提取最终文本

### `core/summary/asr_runtime.py`

运行时准备层，负责：

- 检查当前 Python 环境是否已安装 ASR 依赖
- 必要时自动安装 `requirements-asr.txt`
- 管理 ASR 模型下载状态和持久化状态

### `core/summary/asr_worker.py`

ASR 子进程执行层，负责：

- 加载 FunASR ASR/VAD 模型并执行音频转写
- 保留可读的 `plain_text`
- 在模型返回词级 timestamp 时生成约 25 秒一段的 `segments`
- 将带 `[mm:ss-mm:ss]` 的分段文本写入 `text`，供专业 Markdown 总结提取时间点

### `core/summary/prompts.py`

提示词层，负责：

- 提供默认总结系统提示词
- 提供口语 / 新闻 / 笔记三种总结模板；自动模式在运行时根据信息密度路由到合适模板
- 提供视觉兜底判断和视觉分析提示词
- 提供总结问答提示词，将基础总结作为当前引用内容上下文补充，同时允许通用知识类回答
- 作为插件唯一内置提示词来源，WebUI 不再暴露自定义提示词入口

## 3. 请求流

### 3.1 普通总结流程

1. `main.py` 接收消息事件。
2. 判断是否命中自动 / 口语 / 新闻 / 笔记任一总结命令，并记录本次请求的总结模式。
3. 用 `permissions` 做访问控制。
4. 从引用消息中提取文字、图片、视频源和合并转发 ID，并去重；没有可总结引用内容时流程结束。
5. 确认存在候选内容后，如果消息文本中除总结命令外还有内容，会提取为可选的 `user_hint`。
6. 下载远端视频和图片到 `cache_dir/downloads/`，本地图片路径只引用不清理。
7. 把候选元数据交给 `AISummaryManager`。
8. `AISummaryManager` 汇总引用文字、合并转发集合展开文本、图片输入和视频 ASR；仅在包含视频时准备本地 ASR 运行时，并按需要做视频视觉兜底。
9. 总结 prompt 会携带可选 `user_hint`、引用文字 / 语音转写和视觉内容；当 ASR 返回 timestamp 时，转写内容会带时间段，方便最终笔记总结保留关键时间点。
10. 生成总结并保存为同一私聊或群聊下的临时问答知识库。
11. 每条总结独立回发，并在回复内容里附带 `问答ID` 标记；后续引用该 AI 回复时会优先选择该记录。

合并转发展开只读取第一层平铺节点，并保留节点顺序和发送者标签，例如 `合并转发[1] > 消息[2]`。节点中再次包含的子转发不会继续展开；默认最多展开 100 条第一层节点、保留 16 张图片和 4 个视频；超出部分会在输入文本中标记截断，避免超长转发压垮上下文或下载任务。

### 3.2 总结问答流程

1. 私聊和群聊都只按引用触发：必须引用插件的总结或问答回复，引用消息中的正文直接作为问题。
2. 发送 `qa.exit_commands` 中任一命令会结束本轮问答，但保留知识库记录和引用绑定；发送 `qa.clear_commands` 中任一命令会删除当前 scope 的知识库记录和上下文绑定。
3. 插件只接受引用绑定的 AI 回复作为问答入口，避免多条总结并存时自动串错上下文。
4. 每条插件总结或问答回复都会附带可解析的 `问答ID` 标记，并尽量记录平台返回的消息 ID 作为额外绑定；用户引用哪条 AI 回复，就优先读取那条回复绑定的总结。
5. 如果平台不返回消息 ID，插件仍会从引用消息内容中的 `问答ID` 解析记录；如果记录已过期或被清理，则返回缺失提示。
6. 权限检查复用 `permissions`。
7. 插件按私聊发送者或群号生成 scope，并清理已超过空闲 TTL 的记录。
8. 引用绑定记录时读取对应总结记录。
9. 读取成功会刷新 `last_accessed_at`，后续清理按这次检索时间重新计算。
10. `AISummaryManager.answer_summary_question()` 把基础总结、最近问答和用户问题发送给 LLM。
11. 如果问题询问引用内容事实而基础总结没有覆盖，提示词要求模型说明未覆盖；如果问题询问通用知识、制作方法、工具建议或背景解释，模型可以使用通用知识回答，并区分通用判断和总结中可确认的信息。
12. LLM 返回答案后，插件把本轮用户问题和 AI 回答作为一个问答 pair 追加到对应记录，并按 `qa.history_turns` 保留最近 N 轮。

### 3.3 管理员连通性测试流程

1. 消息内容等于 `admin.test_keyword` 时才进入测试流程，默认值为 `aiping`。
2. 再校验消息是否来自私聊，并确认发送者是否等于 `permissions.admin_id`。
3. 调用 `AISummaryManager.test_llm_connectivity()`。
4. 发送轻量 `ping` 请求到当前 LLM 配置。
5. 将成功或失败结果回发给管理员。

```mermaid
sequenceDiagram
    participant U as User
    participant P as AISummaryPlugin
    participant M as AISummaryManager
    participant L as LLMClient
    U->>P: admin.test_keyword
    P->>P: 校验私聊和 permissions.admin_id
    P->>M: test_llm_connectivity()
    M->>L: complete(ping payload)
    L-->>M: text response / error
    M-->>P: result
    P-->>U: 连通性结果
```

## 4. 配置结构

配置由 `parse_config()` 统一解析，主要分为以下几组：

### `llm`

- `provider_source`：大模型提供商来源，下拉选择 AstrBot 内置提供商或插件自定义提供商
- `persona.enable`：是否启用总结基础人设；关闭时即使 `persona.persona_id` 已选择也不会叠加人设
- `persona.persona_id`：选择 AstrBot 中已配置的人设；启用后会作为最终总结和总结问答的基础 system prompt
- `astrbot_provider.provider_id`：选择 AstrBot 中的 LLM Provider；留空时尝试使用当前会话 AI
- `custom_provider.provider`：插件自定义接口的厂商类型
- `custom_provider.base_url`：插件自定义接口地址
- `custom_provider.api_key`：插件自定义接口密钥
- `custom_provider.model`：插件自定义接口模型名
- `custom_provider.api_version`：插件自定义接口版本参数

### `permissions`

- `admin_id`：管理员账号
- `whitelist` / `blacklist`：群组和用户访问控制

### `trigger`

- `reply_keyword_trigger`：是否开启“引用 + 命令”触发
- `auto_keywords`：自动总结命令列表，默认 `总结一下` / `总结视频`
- `brief_keywords`：简略总结命令列表，默认 `简略总结` / `简单总结`
- `professional_keywords`：专业总结命令列表，默认 `专业总结` / `详细总结`

### `qa`

- `enable`：是否启用总结问答，默认开启
- `exit_commands`：终止问答命令列表，默认 `结束` / `退出`，只结束本轮问答
- `clear_commands`：清理记忆命令列表，默认 `清理` / `清空`，删除当前 scope 的问答知识库
- `record_ttl_minutes`：知识库按最后检索时间保留的分钟数，默认 `30`；设为 `0` 表示不自动清理
- `history_turns`：每个视频记录保留的短问答记忆轮数，默认 `5`；设为 `0` 表示不保留问答历史

### `基础质量`

- `summary.max_completion_tokens`：总结最大输出长度
- `summary.temperature`：总结随机性
- `summary.max_transcript_chars`：进入总结提示词的转写文本上限
- `vision.max_frames`：每个视频最大抽帧数，`0` 表示关闭视觉帧分析
- `vision.frame_size`：视觉帧宽度，`原始尺寸` 表示不做本地缩放
- `vision.image_detail`：发送给视觉模型的图片细节模式
- `vision.batch_size`：每批视觉请求包含的图片数量

### `高级质量`

- `summary.max_concurrent`：总结并发上限，同时约束跨消息总结和单条消息内多候选总结
- `summary.request_timeout_seconds`：总结请求超时时间
- `vision.max_concurrent`：视觉识别批次并发上限
- `vision.jpeg_quality`：ffmpeg 输出 JPEG 的压缩质量
- `vision.max_chars`：视觉观察文本进入最终总结前的长度上限
- `vision.request_timeout_seconds`：视觉识别请求超时时间
- `asr.device`：本地 ASR 推理设备
- `asr.max_concurrent`：语音转写子进程并发上限
- `asr.batch_size_s`：每批转写时长
- `asr.sample_rate`：音频采样率
- `asr.asr_timeout_seconds`：音频提取和转写超时时间

### 内置提示词

总结、视觉兜底判断和视觉分析提示词由 `core/summary/prompts.py` 统一提供，不再通过 WebUI 暴露自定义提示词入口。配置了 `llm.persona.enable` 且 `llm.persona.persona_id` 不为空时，插件会从 AstrBot `Context.persona_manager` 读取对应人设的 system prompt，并只在最终总结和总结问答的用户可见 LLM 请求中作为基础 system prompt 前置；视觉判定、视觉观察、格式修复和连通性测试继续使用各自的中立任务提示。

内置模板变量：

- `{user_hint}`：用户在触发命令之外附加的提示
- `{transcript}`：语音转写文本
- `{visual}`：视觉兜底判定和视觉观察
- `{frame_notes}`：视觉分析抽样帧说明，仅视觉分析提示词使用
- `{summary}`：总结问答的基础总结文本，仅问答提示词使用
- `{history}`：总结问答的最近问答文本，仅问答提示词使用
- `{question}`：总结问答的用户问题，仅问答提示词使用

### `output`

- `status_message`：是否发送处理中提示
- `summary_format`：最终总结内容格式，可选纯文本或 Markdown
- `send_format`：最终总结发送格式，可选文本或图片；图片模式通过 Pillow 和插件本地字体生成温和浅色单列卡片图片
- `qa_answer_format`：最终问答回答格式，可选纯文本或 Markdown
- `qa_send_format`：最终问答发送格式，可选文本或图片；图片模式会把 `问答ID` 作为同条消息里的文本标记一起发送
- `image_font_size`：图片发送模式使用的正文基础字号，建议范围 16-48，标题和小标题按比例放大
- `show_error`：是否回显失败原因
- `enable_summary_repair`：是否在最终回复前启用格式修复，默认清理原始转写泄漏、非标准核对标记和不一致的不确定性注释

### `admin`

- `debug_mode`：调试日志
- `test_keyword`：管理员连通性测试关键词，默认 `aiping`；仅 `permissions.admin_id` 对应账号在私聊中发送时生效，不触发内容总结

## 5. 运行时状态与存储

插件默认把运行数据放在 AstrBot 的插件数据目录下：

- `cache_dir`：插件根缓存目录
- `cache_dir/runtime/`：运行时状态目录
- `cache_dir/downloads/`：远端视频、图片下载目录
- `cache_dir/runtime/images/`：图片发送模式生成的临时总结和问答图片目录
- `cache_dir/runtime/summary_tmp/`：临时音频、转写和视觉分析目录
- `cache_dir/runtime/qa_records/`：总结问答临时知识库记录目录

运行时状态主要包括：

- ASR 依赖检查状态
- ASR 模型下载状态
- 临时安装锁
- 失败原因记录
- 总结问答记录、最近问答和最后检索时间

清理策略：

- 下载文件只在 `downloads/` 内清理
- 临时目录由 `TemporaryDirectory` 自动回收
- 总结问答记录按 `qa.record_ttl_minutes` 和 `last_accessed_at` 自动清理，默认 30 分钟未检索即删除；记录内问答历史按 `qa.history_turns` 裁剪
- 运行状态文件用于重载后恢复进度

## 6. 外部依赖

### 运行时依赖

- AstrBot API
- `aiohttp`
- `ffmpeg`
- Pillow：仅图片发送模式需要；直接执行文本转图片，不依赖浏览器截图环境

### ASR 相关依赖

- `funasr`
- `modelscope`
- `torch`
- `torchaudio`

### LLM 相关依赖

当 `llm.provider_source` 选择 AstrBot 内置提供商时，插件通过 AstrBot `Context.llm_generate()` 调用 `llm.astrbot_provider.provider_id` 指定的 Provider；未指定时尝试使用当前会话 Provider。`llm.persona` 独立于 Provider 来源，启用后会把选中人设解析为 system prompt 并前置到最终总结和问答请求中。

当 `llm.provider_source` 选择插件自定义提供商时，由 `LLMClient` 按 `llm.custom_provider` 中的厂商配置构造请求，当前支持的协议分支包括：

- OpenAI 兼容
- Azure OpenAI
- Anthropic
- Gemini
- Ollama

## 7. 失败边界

系统把失败尽量限制在局部步骤中，而不是让整条链路黑掉：

- 配置缺失时，LLM 连接测试和总结请求会直接报缺项
- ASR 依赖未就绪时，会先进入后台准备
- 本地视频下载失败时，单个候选会标记错误，不影响其他候选
- 转写为空时，可继续走视觉兜底
- LLM 请求失败时，会把错误写回结果消息

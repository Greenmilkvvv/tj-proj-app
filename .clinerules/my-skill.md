# 软著申请资料生成 Skill

当我要求生成软件著作权（软著）相关材料时，必须严格遵循此规则。

## 路径映射

- 原 Skill 目录：`skills/software-copyright-materials/`
- 原 `${CLAUDE_SKILL_DIR}` 在此项目中对应：`skills/software-copyright-materials`
- 所有 Python 脚本执行必须使用：`py -3.10 skills/software-copyright-materials/scripts/<脚本名>.py`
- 参考文档目录：`skills/software-copyright-materials/references/`
- 完整规则原文位于：`skills/software-copyright-materials/SKILL.md`（执行前必须读取作为补充参考）

## 核心原则

- 固定输出目录：当前工作目录下的 `软件著作权申请资料/`，不要写到临时目录
- 先生成 Markdown 草稿，用户确认后再生成正式 Word/TXT
- 正式 Word/TXT 只能写入 `软件著作权申请资料/正式资料/`
- 正式资料文字使用默认黑色字体，不生成彩色文字或超链接
- 代码材料必须来自真实项目源码，禁止 AI 编造代码
- 正式资料中的软件名称和版本号必须与 `草稿/申请表信息.md` 一致
- 脚本只能收集证据和生成文件；行业判断、功能抽取、代码选择、操作手册结构必须由我阅读项目后决定
- 优先抽取前端代码：入口、路由、页面、核心组件、接口封装、状态管理、工具函数

## 代码分页规则

- 每页默认 50 行
- 总页数 >= 60：只输出前 30 页和后 30 页（`代码-前30页.md` + `代码-后30页.md`）
- 总页数 < 60 且候选源码已用尽：生成 `代码-全部.md`
- 总页数 < 60 但候选清单还有可补充源码：停止，要求用户补充选择
- 不为大项目生成全量备份 Word

## 强制人工门禁

以下阶段必须停止等待用户确认，不得自动继续：

1. **environment**：DOCX 环境缺失时，用户必须选择"安装完整环境"或"使用基础 DOCX"
2. **project**：存在多个候选项目时，用户必须指定
3. **business**：`草稿/业务理解.md` 生成后，用户确认行业、用户、功能
4. **application-fields**：`草稿/申请表信息.md` 生成后，用户补全硬件/系统环境、著作权人等
5. **code-selection**：`草稿/代码文件选择.json` 生成后，用户确认抽取文件
6. **screenshot-method**：用户必须在 Chrome DevTools MCP / Codex Computer Use / 自行截图 / skip 中选择
7. **markdown**：全部 Markdown 草稿完成后，用户确认进入 Word/TXT 生成

记录门禁命令：
```bash
py -3.10 skills/software-copyright-materials/scripts/confirm_stage.py --workdir 软件著作权申请资料 --stage <阶段名> --note "<用户确认内容>"
```

## 完整工作流

### 1. 启动环境检查

```bash
py -3.10 skills/software-copyright-materials/scripts/check_environment.py --out-dir 软件著作权申请资料
```

输出 `环境检查.md` + `环境检查.json`。告知用户 DOCX 环境状态，缺失时等待用户选择（安装完整环境 / 基础兜底）。记录 `environment` 门禁。

### 2. 定位项目

扫描当前目录，避开 skill 自身、输出目录、node_modules、构建产物。多个候选时停止询问。

### 3. 分析项目

```bash
py -3.10 skills/software-copyright-materials/scripts/analyze_project.py --project <项目目录> --out 软件著作权申请资料/analysis/project.json
```

### 4. 形成业务理解

先收集证据：
```bash
py -3.10 skills/software-copyright-materials/scripts/generate_business_context.py --project <项目目录> --analysis 软件著作权申请资料/analysis/project.json --software-name "<软件全称>" --out-dir 软件著作权申请资料/草稿
```

输出 `业务理解证据.md` + `业务理解证据.json` + `业务理解模型稿模板.json`。

然后我必须自行阅读证据和项目源码，判断：行业/领域、目标用户、核心价值、业务功能、操作流程、操作手册结构。生成业务理解 JSON 包含：product_positioning, industry, target_users, core_value, business_features, business_feature_details, operation_flow, application_purpose, main_functions, technical_characteristics, manual_sections, manual_modules, system_requirements, faq, glossary。

每个 manual_module 必须含：title, evidence, purpose, usage/usage_scenario, entry, visible_elements, operation_steps, validation_rules, feedback, screenshot。

再运行带 model-context 的命令生成最终 `业务理解.md/json`。等待用户确认后记录 `business` 门禁。

### 5. 引导用户确认申请表字段

必填字段（按官网表单顺序）：软件全称、软件简称（可选）、版本号（项目版本号<V1.0时明确询问是否写V1.0）、软件分类（默认应用软件）、开发完成日期(YYYY-MM-DD)、开发方式（默认单独开发）、软件说明（默认原创）、发表状态、首次发表日期（已发表时）、著作权人（国家/省市/类型/姓名/证件类型/证件号）、权利范围、权利取得方式、开发硬件环境(≤50字符)、运行硬件环境(≤50字符)、开发操作系统(≤50字符)、开发环境/开发工具(≤50字符，格式: 开发环境: xxx/开发工具: xxx)、运行平台/操作系统(≤50字符)、支撑环境/支持软件(≤50字符)、编程语言(≤120字符)、源程序量(纯数字行数)、开发目的(≤50字符)、面向领域/行业(≤50字符)、主要功能(500~1300字符)、技术特点(标签+文本≤100字符)、页数。

硬件/系统环境优先读取当前电脑配置作为建议值。项目可推断字段给建议值。

### 6. 确认代码文件选择

生成候选清单：
```bash
py -3.10 skills/software-copyright-materials/scripts/propose_code_selection.py --project <项目目录> --analysis 软件著作权申请资料/analysis/project.json --out-dir 软件著作权申请资料/草稿
```

我必须阅读业务理解、候选文件、入口文件后判断哪些源码最能体现软件真实功能，修改 `代码文件选择.json`（selected: true/false + model_reason）。优先前端代码，不足 60 页再补充后端。用户确认后记录 `code-selection` 门禁。

### 7. 生成 Markdown 草稿

**代码材料：**
```bash
py -3.10 skills/software-copyright-materials/scripts/extract_code_material.py --project <项目目录> --analysis 软件著作权申请资料/analysis/project.json --selection 软件著作权申请资料/草稿/代码文件选择.json --software-name "<软件全称>" --version "<版本号>" --out-dir 软件著作权申请资料/草稿
```

**申请表信息：**
```bash
py -3.10 skills/software-copyright-materials/scripts/generate_application_info.py --analysis 软件著作权申请资料/analysis/project.json --code-manifest 软件著作权申请资料/草稿/代码提取清单.json --business-context 软件著作权申请资料/草稿/业务理解.json --software-name "<软件全称>" --version "<版本号>" --out-dir 软件著作权申请资料/草稿
```

需停止让用户检查补全后再记录 `application-fields` 门禁。

**操作手册草稿：**
```bash
py -3.10 skills/software-copyright-materials/scripts/generate_manual_draft.py --analysis 软件著作权申请资料/analysis/project.json --business-context 软件著作权申请资料/草稿/业务理解.json --software-name "<软件全称>" --version "<版本号>" --out-dir 软件著作权申请资料/草稿
```

操作手册要求：
- 一级章节标题用中文大写序号：`一、相关文档`、`二、说明`...
- 骨架：相关文档(表格) → 说明 → 功能特点 → 系统要求(表格) → 按页面/流程逐章操作 → 常见问题解答 → 术语表
- 功能特点每项用段落展开，不得用项目符号列表
- 每个核心页面写清：使用场景、页面用途、进入位置、可见内容、用户动作、输入限制/异常提示、操作结果、截图预留
- 语言面向普通用户，不写代码实现、框架、接口、状态管理
- 禁止使用"旨在、赋能、一站式、智能化、高效便捷"等套话
- 必须同步输出 `操作手册自检记录.md/json`，至少3轮自检
- 完整草稿完成后只让用户做一次整体确认

### 8. 截图处理

先让用户选择方式（Chrome DevTools MCP / Codex Computer Use / 自行截图 / skip），记录 `screenshot-method` 门禁。
- 选 skip 时操作手册保留可见截图预留文字如：`【截图预留：请在此处插入"XXX"页面截图。】`
- 自行截图时创建 `软件著作权申请资料/用户截图/`，用户放入后运行：
```bash
py -3.10 skills/software-copyright-materials/scripts/capture_screenshots.py --manual-dir 软件著作权申请资料/用户截图 --out-dir 软件著作权申请资料/截图
```

### 9. 用户确认 Markdown

检查：软件名称和版本号一致性、代码材料页眉、操作手册页眉、业务理解准确性、申请表字段完整性、代码真实性、操作手册可读性、截图。记录 `markdown` 门禁。

### 10. 生成正式 Word/TXT

```bash
py -3.10 skills/software-copyright-materials/scripts/build_docx_from_md.py --workdir 软件著作权申请资料 --software-name "<软件全称>" --version "<版本号>"
```

输出：
- `正式资料/申请表信息.txt`
- `正式资料/<软件全称>-代码(前30页).docx` + `正式资料/<软件全称>-代码(后30页).docx`（>=60页时）
- `正式资料/<软件全称>-代码(全部).docx`（<60页时）
- `正式资料/<软件全称>_操作手册.docx`
- `正式资料/生成报告.md`

操作手册页眉格式：左侧"软件全称 版本号"，右侧"第 X 页"。

### 11. 三轮验证

1. 文件完整性：Word/TXT 存在且非空
2. 代码真实性：抽样回溯项目源码
3. 业务真实性：回溯到业务理解.md 和项目文档
4. 一致性和格式：软件名称、版本号、页数、字段、标题、截图引用

## 申请表示例字段口径

- 软件全称：由用户确认，所有正式资料以此为准
- 版本号：由用户确认，优先读取项目配置；< V1.0 时必须明确询问是否写 V1.0
- 软件开发环境/开发工具：`开发环境: Windows 11/开发工具: Visual Studio Code`，不写 React、Vite 等技术栈
- 开发操作系统：开发电脑的操作系统
- 运行平台/操作系统：软件运行所在环境
- 支撑环境/支持软件：直接列出依赖（如 Node.js、浏览器），不加前缀
- 源程序量：纯数字行数（不含"行"字）
- 开发目的：一句话说明（≤50字符）
- 主要功能：500~1300字符详细描述
- 技术特点：多选标签（APP/游戏软件/教育软件/金融软件/医疗软件/地理信息软件/云计算软件/信息安全软件/大数据软件/人工智能软件/VR软件/5G软件/小程序/物联网软件/智慧城市软件）+ 文本描述≤100字符

## 参考文档

执行对应阶段时应读取以下文件获取详细规则：
- `skills/software-copyright-materials/references/copyright_material_rules.md` — 代码分页和鉴别材料规则
- `skills/software-copyright-materials/references/manual_structure.md` — 操作手册写作规范（结构、口径、AI味检查）
- `skills/software-copyright-materials/references/application_fields.md` — 申请表字段详细说明
- `skills/software-copyright-materials/references/code_selection_rules.md` — 代码选择规则
- `skills/software-copyright-materials/references/business_understanding_rules.md` — 业务理解规则
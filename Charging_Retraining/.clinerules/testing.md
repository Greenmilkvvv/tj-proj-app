# Python 执行规则

**Attention: ** 不要再修改功能文档

【强制】执行 Python 脚本时，必须使用 `py -3.10` 命令，而不是 `python`。
原因：系统默认的 `python` 命令不可用。

正确示例：
py -3.10 script.py

禁止使用：
python script.py

【背景】
你的系统同时安装了 Python 3.9 和 3.10，默认的 python 命令指向了不可用的版本。
py -3.10 是唯一可靠的方式。
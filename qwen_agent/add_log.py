# 遍历本目录所有py文件，在开始时添加一行代码：from qwen_agent.log_util import log_execution
import os
import re
from pathlib import Path

TARGET_IMPORT = "from qwen_agent.log_util import log_execution"
TARGET_DIR = Path(__file__).parent


def should_skip_file(filepath: Path) -> bool:
    """判断是否应该跳过该文件"""
    # 跳过自身
    if "add_log.py" in filepath.name or "log_util.py" in filepath.name:
        return True
    # 跳过 __init__.py（可选，根据需求调整）
    # if filepath.name == "__init__.py":
    #     return True
    return False


def has_import_statement(content: str, import_stmt: str) -> bool:
    """检查文件中是否已存在指定的 import 语句"""
    # 转义特殊字符进行精确匹配
    pattern = re.escape(import_stmt)
    return bool(re.search(pattern, content))


def add_import_to_file(filepath: Path, import_stmt: str) -> bool:
    """
    向文件添加 import 语句
    返回是否成功添加
    """
    try:
        # 读取文件内容
        with open(filepath, "r", encoding="utf-8") as f:
            original_content = f.read()

        # 检查是否已存在该 import
        if has_import_statement(original_content, import_stmt):
            print(f"⏭️  跳过 (已存在): {filepath.name}")
            return False

        lines = original_content.splitlines(keepends=True)

        # 找到插入位置
        insert_position = 0

        # 如果文件以 shebang 开头，跳过 shebang 行
        if lines and lines[0].startswith("#!"):
            insert_position = 1

        # 跳过编码声明
        if insert_position < len(lines) and re.match(r"^#.*coding[:=]", lines[insert_position]):
            insert_position += 1

        # 处理 docstring
        remaining_content = "".join(lines[insert_position:])
        stripped = remaining_content.lstrip()

        # 检查是否有 docstring
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote_type = stripped[:3]
            # 查找 docstring 结束位置
            end_pos = stripped.find(quote_type, 3)
            if end_pos != -1:
                # 找到 docstring 结束，计算在原内容中的位置
                docstring_end_in_remaining = end_pos + 3
                # 找到在 remaining_content 中的实际位置
                actual_pos = 0
                char_count = 0
                for i, line in enumerate(lines[insert_position:]):
                    if char_count >= docstring_end_in_remaining:
                        insert_position += i
                        break
                    char_count += len(line)
                else:
                    # docstring 可能跨越多行但未找到结束标记
                    pass

        # 在插入位置添加空行和 import 语句
        new_line = "\n" if lines else ""
        import_line = f"{import_stmt}\n"

        # 如果插入位置前有内容，确保有空行分隔
        if insert_position > 0 and insert_position < len(lines):
            lines.insert(insert_position, new_line)
            lines.insert(insert_position + 1, import_line)
        else:
            lines.insert(insert_position, import_line)

        new_content = "".join(lines)

        # 写回文件
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)

        print(f"✅ 已添加: {filepath.name}")
        return True

    except Exception as e:
        print(f"❌ 处理失败 {filepath.name}: {e}")
        return False


def main(dry_run: bool = False):
    """
    遍历目录下所有 Python 文件并添加 import 语句

    Args:
        dry_run: 如果为 True，只显示将要执行的操作，不实际修改文件
    """
    py_files = list(TARGET_DIR.glob("**/*.py"))

    if not py_files:
        print("未找到 Python 文件")
        return

    modified_count = 0
    skipped_count = 0

    for filepath in sorted(py_files):
        if should_skip_file(filepath):
            print(f"⏭️  跳过 (系统文件): {filepath.name}")
            skipped_count += 1
            continue

        if dry_run:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            if not has_import_statement(content, TARGET_IMPORT):
                print(f"📝 将会添加: {filepath.name}")
                modified_count += 1
            else:
                print(f"⏭️  跳过 (已存在): {filepath.name}")
                skipped_count += 1
        else:
            if add_import_to_file(filepath, TARGET_IMPORT):
                modified_count += 1
            else:
                skipped_count += 1

    print(f"\n{'='*50}")
    print(f"处理完成!")
    print(f"修改: {modified_count} 个文件")
    print(f"跳过: {skipped_count} 个文件")
    print(f"{'='*50}")


if __name__ == "__main__":
    import sys

    # 支持 --dry-run 参数进行预览
    is_dry_run = "--dry-run" in sys.argv

    if is_dry_run:
        print("🔍 预览模式 - 以下文件将被修改:\n")

    main(dry_run=is_dry_run)

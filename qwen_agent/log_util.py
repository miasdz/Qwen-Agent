from loguru import logger
import time
from functools import wraps
import sys
import contextvars

# 1. 移除 Loguru 默认的处理器
logger.remove()

# 2. 配置全局默认的 extra 值，这是必须的，可以防止 KeyError
logger.configure(extra={"request_id": "N/A"})

# 3. 添加新的处理器，并在格式字符串中使用 {extra[request_id]}
console_log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<magenta>request_id: {extra[request_id]}</magenta> | " # 这里引用 extra 字段
    "<level>{message}</level>"
)

logger.add(
    sys.stdout,
    format=console_log_format,
    level="DEBUG",
    colorize=True,
    backtrace=True,
    diagnose=True,
    enqueue=True
)

file_log_format = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "request_id: {extra[request_id]} | "
    "{message}"
)
logger.add(
    "logs/run_{time:YYYY-MM-DD}.log",
    format=file_log_format,
    level="DEBUG",
    colorize=True,
    backtrace=True,
    diagnose=True,
    enqueue=True
)

stack = contextvars.ContextVar('user', default='root')

def log_execution(func):
    import inspect

    # 如果是生成器函数，直接返回原函数，不进行包装
    # 因为 Gradio 等框架需要直接控制生成器的迭代
    if inspect.isgeneratorfunction(func):
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        # 使用 __qualname__ 获取限定名称（包含类名），用 __name__ 作为备选
        func_name = getattr(func, '__qualname__', func.__name__)
        module_name = getattr(func, '__module__', '')

        # 组合完整的调用路径，例如: "my_module.MyClass.my_func"
        full_name = f"{module_name}.{func_name}" if module_name else func_name

        # 记录入参
        logger.debug(f"⏳ 开始执行: {full_name}, 参数: {args}, {kwargs}")

        start_time = time.time()

        # 对于普通函数（包括 __init__），正常执行并记录日志
        result = None
        exception_occurred = False

        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            exception_occurred = True
            duration = time.time() - start_time
            logger.error(f"❌ 执行失败: {full_name}, 耗时: {duration:.4f}秒, 错误: {e}")
            raise
        finally:
            if not exception_occurred:
                # 记录耗时（仅在无异常时）
                duration = time.time() - start_time
                if func.__name__ == '__init__':
                    logger.debug(f"✅ 初始化完成: {full_name}, 耗时: {duration:.4f}秒")
                else:
                    logger.debug(f"✅ 执行完毕: {func.__name__}, 耗时: {duration:.4f}秒, 返回值：{result}")

    return wrapper

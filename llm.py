from loguru import logger
import config


def cost_time(func):
    import time

    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        logger.debug(f"func {func.__name__} cost time:{time.perf_counter() - t:.0f} s")
        return result

    return fun


class LLM:
    system_prompt = config.system_prompt
    max_length = config.max_length
    max_new_tokens = config.max_new_tokens
    max_time = config.max_time

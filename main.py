import sys
from codet5p import CodeT5p
from starcoder import StarCoder
from secgpt import SecGPT
from eval_llm import eval_llm
from loguru import logger


def main():
    if len(sys.argv) < 2:
        print("Usage: ppython script.py [model_type]")
        sys.exit(1)

    model_type = sys.argv[1].lower()

    if model_type == "codet5p":
        model = CodeT5p()
    elif model_type == "starcoder":
        model = StarCoder()
    elif model_type == "secgpt":
        model = SecGPT()
    else:
        print("Unsupported model type")
        sys.exit(1)

    logger.remove()  # remove default handler
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        # "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "{extra[model_type]} - <level>{message}</level>"
    )
    logger.configure(extra={"model_type": model_type})
    logger.add(sys.stdout, format=log_format)
    logger.add(model_type + ".txt", format=log_format)

    # print(model.run_model("write a python function of quick sort."))
    eval_llm(model, model_type)


if __name__ == "__main__":
    main()

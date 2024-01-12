import os
from loguru import logger


def list_files(directory):
    list = []
    for name in os.listdir(directory):
        list.append(name)
    return list


def open_the_only_file(directory):
    # open the only poc file of the directory
    files_and_dirs = os.listdir(directory)

    files = [
        file for file in files_and_dirs if os.path.isfile(os.path.join(directory, file))
    ]

    if files:
        first_file_path = os.path.join(directory, files[0])
        with open(first_file_path, "r") as file:
            return file.read()
    else:
        logger.warning("poc not found: " + directory)
        return "error"


def eval_llm(model, model_type):
    cve_list = list_files("dataset/python/")
    # handle each cv
    for cve in cve_list:
        logger.info("Handle: " + cve)
        for exp in ["1", "2", "3"]:
            save_dir = os.path.join("./result", model_type, exp)
            target = os.path.join(save_dir, cve + ".ruby")
            if os.path.exists(target):
                logger.info(target + " exist!")
                continue

            poc = open_the_only_file(os.path.join("dataset/python/", cve, exp))
            if poc != "error":
                model_output = model.run_model(poc)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                with open(target, "w") as f:
                    f.write(model_output)
                    logger.info("Done: " + target)

import sys
from codet5p import CodeT5p

from starcoder import StarCoder


def main():
    # if len(sys.argv) < 3:
    #     print("Usage: python script.py [model_type] [input_text]")
    #     sys.exit(1)

    model_type = sys.argv[1].lower()
    # input_text = sys.argv[2]

    if model_type == "codet5p":
        model = CodeT5p()
    elif model_type == "starcoder":
        model = StarCoder()
    else:
        print("Unsupported model type")
        sys.exit(1)

    print(model.run_model("write a python function of quick sort."))
    print("-----------------------------------")
    print("-----------------------------------")
    print(model.run_model("write a C function of quick sort."))


if __name__ == "__main__":
    main()

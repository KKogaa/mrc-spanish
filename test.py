import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="test", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-m", "--model", help="model name")
    parser.add_argument("-v", "--verbose", help="project name")
    parser.add_argument("-v", "--verbose", help="archive mode")
    args = parser.parse_args()
    config = vars(args)
    print(config)

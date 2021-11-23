class STYLE:
    BOLD = '\033[1m'
    END = '\033[0m'


def bold(text):
    return STYLE.BOLD + text + STYLE.END

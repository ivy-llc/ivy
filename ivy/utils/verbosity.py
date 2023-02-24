# TODO should this still be here?
import termcolor

level = 0


def cprint(message, color="green"):
    print(termcolor.colored(message, color))

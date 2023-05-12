from ._checks import check_type, check_value
from .logs import logger


def query_yes_no(question, default="yes", *, retries=5):
    """Ask a yes/no question via input() and return their answer.

    Parameters
    ----------
    question : str
        Question presented to the user.
    default : str
        Key correspoinding to the presumed answer if the user just hits <Enter>. If
        None, an answer is required from the user.
    retries : int
        Number of retries for a given input until an error is raised.

    Returns
    -------
    answer : bool
        True for "yes" or False for "no".
    """
    check_type(question, (str,), item_name="question")
    check_type(default, (str, None), item_name="default")
    if default is not None:
        check_value(default, ("yes", "no"), item_name="default")
    check_type(retries, (int,), item_name="retries")

    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}

    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "

    attempt = 0
    while True:
        attempt += 1
        if attempt > retries:
            raise ValueError

        choice = input("[IN] " + question + prompt).lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            logger.warning("Please respond with 'yes'/'y' or 'no'/'n'.")

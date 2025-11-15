
# I N I T I A L   P A R A M E T E R S
# -----------------------------------

HEADER = '\033[95m'     # header
BLUE = '\033[94m'       # blue
GREEN = '\033[92m'      # green
YELLOW = '\033[93m'     # yellow
RED = '\033[91m'        # red
BOLD = '\033[1m'        # bold
UNDERLINE = '\033[4m'   # underline
ENDC = '\033[0m'        # end command


# F U N C T I O N S
# -----------------------------------
def printc(text, alarm):
    """
    Colored message.
    Parameters
    ----------
    text:   [str] message to display.
    alarm:  [str] alarm type to define color.
    ----------
    """

    alarm_colors = [HEADER, BLUE, GREEN, YELLOW, RED, BOLD, UNDERLINE]
    alarm_types = ['title1', 'info', 'ok', 'warn', 'fail', 'title2', 'title3']
    
    try:
        idx_color = alarm_types.index(alarm)
        print(f'{alarm_colors[idx_color]}{text}{ENDC}')

    except ValueError:
        print(text)
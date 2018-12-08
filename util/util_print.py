class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


str_stage = bcolors.OKBLUE + '==>' + bcolors.ENDC
str_verbose = bcolors.OKGREEN + '[Verbose]' + bcolors.ENDC
str_warning = bcolors.WARNING + '[Warning]' + bcolors.ENDC
str_error = bcolors.FAIL + '[Error]' + bcolors.ENDC

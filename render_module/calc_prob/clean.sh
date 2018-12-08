
# ANSI color codes
RS="\033[0m"    # reset
HC="\033[1m"    # hicolor
UL="\033[4m"    # underline
INV="\033[7m"   # inverse background and foreground
FBLK="\033[30m" # foreground black
FRED="\033[31m" # foreground red
FGRN="\033[32m" # foreground green
FYEL="\033[33m" # foreground yellow
FBLE="\033[34m" # foreground blue
FMAG="\033[35m" # foreground magenta
FCYN="\033[36m" # foreground cyan
FWHT="\033[37m" # foreground white
BBLK="\033[40m" # background black
BRED="\033[41m" # background red
BGRN="\033[42m" # background green
BYEL="\033[43m" # background yellow
BBLE="\033[44m" # background blue
BMAG="\033[45m" # background magenta
BCYN="\033[46m" # background cyan
BWHT="\033[47m" # background white

function rm_if_exist() {
if [ -f "$1" ]; then
    rm "$1";
    echo -e "${FGRN}File $1 removed${RS}"
elif [ -d "$1" ]; then
    rm -r "$1";
    echo -e "${FBLE}Directory $1 removed${RS}"
else
    echo -e "${FRED}$1 not found${RS}"
fi
}

rm_if_exist "calc_prob/src/calc_prob_kernel.cu.o"
rm_if_exist "__pycache__"
rm_if_exist "dist"
rm_if_exist "build"
rm_if_exist "pytorch_calc_stop_problility.egg-info"
rm_if_exist ".cache"
rm_if_exist "calc_prob/__pycache__"
rm_if_exist "calc_prob/_ext"
rm_if_exist "calc_prob/functions/__pycache__"
rm_if_exist "calc_prob/modules/__pycache__"

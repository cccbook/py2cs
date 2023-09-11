#SCRIPT_PATH="${BASH_SOURCE:-$0}"
#python ${SCRIPT_PATH}/GptChat/GptChat.py


SCRIPT_PATH="${BASH_SOURCE:-$0}"
#echo "Value of SCRIPT_PATH: ${SCRIPT_PATH}"

ABS_SCRIPT_PATH="$(realpath "${SCRIPT_PATH}")"
#echo "Value of ABS_SCRIPT_PATH: ${ABS_SCRIPT_PATH}"

ABS_DIRECTORY="$(dirname "${ABS_SCRIPT_PATH}")"
#echo "Value of ABS_DIRECTORY: ${ABS_DIRECTORY}"

python ${ABS_DIRECTORY}/shellgpt.py

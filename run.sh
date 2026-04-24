#!/bin/bash

# Запускаем, выводим всё в консоль и параллельно ищем путь к логу
# Файл .tmp_path нужен, чтобы вытащить переменную из потока
python run.py "$@" | tee /dev/tty | grep "Log file:" > .tmp_path

# Извлекаем чистый путь из пойманной строки
# (sed уберет всё лишнее до двоеточия)
LOG_PATH=$(cat .tmp_path | sed 's/.*Log file: //')
rm .tmp_path

echo "------------------------------------------------"
echo "Bash caught the log path: $LOG_PATH"
# Теперь ты можешь, например, открыть его в конце:
# tail -n 20 "$LOG_PATH"
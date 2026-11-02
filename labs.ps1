<#
    Быстрый запуск лаб в PowerShell: lab7, lab12, lab 7, labs.

    Один раз в текущем окне:
        . "C:\Users\Максим\MDKkop\labs.ps1"

    Чтобы работало всегда, допиши эту же строку в свой профиль:
        notepad $PROFILE

    В cmd.exe ничего подключать не нужно — там работают файлы lab7.cmd и т.п.
#>

$script:MdkRoot = $PSScriptRoot

# Список лаб
function labs {
    & python (Join-Path $script:MdkRoot 'run.py') @args
}

# lab 7 [имя_файла] [аргументы]
function lab {
    $number = $args[0]
    $rest = @()
    if ($args.Count -gt 1) { $rest = $args[1..($args.Count - 1)] }
    & python (Join-Path $script:MdkRoot 'run.py') $number @rest
}

function lab2  { lab 2  @args }
function lab3  { lab 3  @args }
function lab4  { lab 4  @args }
function lab5  { lab 5  @args }
function lab6  { lab 6  @args }
function lab7  { lab 7  @args }
function lab8  { lab 8  @args }
function lab9  { lab 9  @args }
function lab10 { lab 10 @args }
function lab11 { lab 11 @args }
function lab12 { lab 12 @args }
function lab13 { lab 13 @args }
function lab14 { lab 14 @args }
function lab15 { lab 15 @args }

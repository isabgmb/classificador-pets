@echo off
rem Roda o projeto inteiro localmente, sempre com o Python do ambiente conda "pets".
rem Uso: dois cliques neste arquivo, ou no PowerShell:  .\rodar_tudo.bat --rapido
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PY=%USERPROFILE%\anaconda3\envs\pets\python.exe"
set "CONDA=%USERPROFILE%\anaconda3\Scripts\conda.exe"

if not exist "%PY%" (
    if not exist "%CONDA%" (
        echo Nao encontrei o Anaconda em %USERPROFILE%\anaconda3.
        echo Instale o Anaconda ou crie o ambiente "pets" com Python 3.12.
        pause
        exit /b 1
    )
    echo Criando o ambiente "pets" com Python 3.12. Isso so acontece uma vez...
    "%CONDA%" create -n pets python=3.12 -y
    if errorlevel 1 (
        echo Falha ao criar o ambiente "pets".
        pause
        exit /b 1
    )
)

"%PY%" rodar_tudo.py %*
set "CODIGO=%ERRORLEVEL%"
if not "%CODIGO%"=="0" (
    echo.
    echo Algo deu errado. Copie a mensagem acima e mande para revisao.
)
pause
exit /b %CODIGO%

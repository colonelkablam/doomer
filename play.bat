@echo off
rem OK Doomer for Windows: double-click to play (or a shortcut to this on
rem the desktop). The first run sets up Python's environment (a minute or
rem two); every run updates the game first, so it matches everyone else's
rem (different versions can't play together), then starts it. If anything
rem goes wrong, the window stays open to show why.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run: setting up, this takes a minute or two...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo.
        echo Couldn't set up: is Python 3.11 or newer installed, from python.org?
        pause
        exit /b 1
    )
)

echo Checking for updates...
git pull --ff-only
if errorlevel 1 echo (Couldn't update: playing the version you have.)

rem Once: an "OK Doomer" shortcut on the desktop, with the game's icon (a .bat
rem can't have one of its own). The marker file stops it coming back if you
rem delete the shortcut; delete .shortcut-made to have it made again.
if exist ".shortcut-made" goto :shortcut_done
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), 'OK Doomer.lnk')); $s.TargetPath = '%~dp0play.bat'; $s.WorkingDirectory = '%~dp0'; $s.IconLocation = '%~dp0ok-doomer.ico'; $s.Save()"
if errorlevel 1 (
    echo (Couldn't make the desktop shortcut: carrying on.)
) else (
    echo Made an "OK Doomer" shortcut on the desktop.
    echo made> ".shortcut-made"
)
:shortcut_done

".venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo.
    echo Couldn't install what the game needs: see above.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m ok_doomer.client %*
if errorlevel 1 pause

@echo off
REM ---------------------------------------------------------------------------
REM  Publish this folder to https://github.com/karemdm/nutrilabel-ocr
REM
REM  Before running: create an EMPTY repository named "nutrilabel-ocr" at
REM  https://github.com/new  (no README, no .gitignore, no licence).
REM  Git opens a browser window to sign in - never paste tokens or passwords here.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0.."

where git >nul 2>&1 || (echo [error] Git is not installed: winget install --id Git.Git -e & exit /b 1)

REM The README shows docs\assets\ui.png: create it if run_all.bat was not run yet.
if exist "docs\assets\ui.png" goto :screenshot_done
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
echo [publish] generating the interface screenshot docs\assets\ui.png
python scripts\make_ui_snapshot.py --engine rapidocr
set "EDGE=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
if not exist "%EDGE%" set "EDGE=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
if exist "%EDGE%" "%EDGE%" --headless=new --disable-gpu --hide-scrollbars --virtual-time-budget=15000 --window-size=1600,1050 --screenshot="%CD%\docs\assets\ui.png" "file:///%CD:\=/%/docs/assets/ui_snapshot.html" >nul 2>&1
:screenshot_done

REM Commits are signed with GitHub's private "noreply" address, so no personal
REM e-mail ends up in the public history. For your exact address (ID+user form) see
REM https://github.com/settings/emails and replace it below if you wish.
if not exist ".git" git init -b main
git config user.name "Karem Marcomini"
git config user.email "72208245+karemdm@users.noreply.github.com"

git add .
git commit -m "nutrilabel-ocr v0.1.0: extractor, 14 OCR engines, benchmark, CLI and web UI"
git remote get-url origin >nul 2>&1 || git remote add origin https://github.com/karemdm/nutrilabel-ocr.git
git branch -M main
git push -u origin main || (echo [error] push failed - does the empty repository exist on GitHub? & exit /b 1)

echo.
echo Published: https://github.com/karemdm/nutrilabel-ocr
echo The "Actions" tab runs the tests automatically.

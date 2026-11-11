@echo off
REM ---------------------------------------------------------------------------
REM  nutrilabel-ocr - run everything on Windows with ONE command:
REM
REM      scripts\run_all.bat            (benchmark on 30 real photos, ~5-10 min)
REM      scripts\run_all.bat full       (benchmark on all 199 test photos)
REM
REM  Steps: install -> tests -> benchmark -> report/charts/gallery ->
REM         demo GIF -> UI screenshot (docs\assets\ui.png)
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0.."

set "LIMIT=--limit 30"
if /i "%~1"=="full" set "LIMIT="

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo [setup] creating virtual environment .venv
    python -m venv .venv || goto :fail
    call ".venv\Scripts\activate.bat"
)
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" set "PATH=%PATH%;C:\Program Files\Tesseract-OCR"

echo.
echo ===== 1/6 Installing the package (rapidocr + tesseract + benchmark tools)
python -m pip install --quiet --upgrade pip
pip install --quiet -e ".[dev,rapidocr,tesseract,benchmark]" || goto :fail

echo.
echo ===== 2/6 Running the tests
pytest -q
if errorlevel 1 echo [warning] some tests failed - see above. Continuing.

echo.
echo ===== 3/6 Benchmark on real Open Food Facts photos (first run downloads ~650 MB)
python -m benchmark.run %LIMIT% || goto :fail

echo.
echo ===== 4/6 Report: table, charts, annotated gallery and GIF of real photos
python -m benchmark.report --gallery 6 || goto :fail

echo.
echo ===== 5/6 Demo GIF for the README
python scripts\make_demo.py --engine rapidocr || goto :fail

echo.
echo ===== 6/6 Screenshot of the web interface
python scripts\make_ui_snapshot.py --engine rapidocr || goto :fail
set "EDGE=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
if not exist "%EDGE%" set "EDGE=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
if exist "%EDGE%" (
    "%EDGE%" --headless=new --disable-gpu --hide-scrollbars --virtual-time-budget=15000 --window-size=1600,1050 ^
        --screenshot="%CD%\docs\assets\ui.png" "file:///%CD:\=/%/docs/assets/ui_snapshot.html" >nul 2>&1
    echo screenshot saved to docs\assets\ui.png
) else (
    echo [warning] Microsoft Edge not found: open docs\assets\ui_snapshot.html and save a screenshot as docs\assets\ui.png
)

echo.
echo ===========================================================================
echo  DONE. Results:
echo ===========================================================================
type docs\benchmark_results.md
echo.
echo  Files updated: docs\benchmark_results.md, docs\assets\*.png, docs\assets\*.gif,
echo                 benchmark\results\summary.json
echo  Next step - publish to GitHub:   scripts\publish.bat
goto :eof

:fail
echo.
echo [error] the step above failed. Copy the messages above and send them to the assistant.
exit /b 1

@echo off
REM Double-click me: first-time setup + launch Isaac Sim GUI
echo [1/2] Running post_install.bat ...
call D:\isaacsim\post_install.bat
echo.
echo [2/2] Launching Isaac Sim. FIRST launch compiles shaders: 10-20 min on this laptop. Do NOT close this window.
echo       If asked to accept the EULA, type Yes and press Enter.
call D:\isaacsim\isaac-sim.bat
pause

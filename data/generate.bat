
@echo on
set home=%cd%
cd %1
call :treeProcess
goto :eof

:treeProcess
rem Do whatever you want here over the files of this subdir, for example:
for %%f in (*.rep) do call :convert "%%~f"
for /D %%d in (*) do (
    cd %%d
    call :treeProcess
    cd ..
)
exit /b

:convert
set curdir=%cd%
cd "%home%"
screp.exe -map=true --outfile "%curdir%"_"%~n1".json "%curdir%"/"%~1"
cd "%curdir%"
exit /b

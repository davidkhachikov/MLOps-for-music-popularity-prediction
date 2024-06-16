@echo off
cd /d "%~dp0.\.."
call.\env\Scripts\activate
pip install -r.\requirements.txt
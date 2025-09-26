@echo off
echo Installing YOLO Krypton dependencies...
echo.

echo Installing customtkinter...
pip install customtkinter
echo.

echo Installing Pillow...
pip install Pillow
echo.

echo Installing numpy...
pip install numpy
echo.

echo Installing opencv-python...
pip install opencv-python
echo.

echo Installing ultralytics (this may take a while)...
pip install ultralytics
echo.

echo Installing matplotlib...
pip install matplotlib
echo.

echo Installing pandas...
pip install pandas
echo.

echo.
echo Installation complete!
echo You can now run: python main.py
pause

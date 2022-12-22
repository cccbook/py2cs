from pyvirtualdisplay import Display
# disp = Display().start()
disp=Display(backend="xvfb")
# display is active
disp.stop()
# display is stopped
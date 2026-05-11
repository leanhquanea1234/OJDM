from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306, ssd1325, ssd1331, sh1106
from time import sleep

serial = i2c(port=1, address=0x3C)
device = ssd1306(serial, rotate=2)
with canvas(device) as draw:
    draw.text((10, 0), "Hello World", fill="white")
    draw.text((10, 50), "Orange Juice", fill="white")
'''
with canvas(device) as draw:
    draw.point([(0, 0), (127, 63)], fill="white")
'''

try:
    while True:
        sleep(1)

except KeyboardInterrupt:
    print("Program terminated by user.")
finally:
    device.clear


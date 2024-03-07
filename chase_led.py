'''
File name: chase_led.py
Author: WU Yuxuan
Created time: 2024.2.6
Description: Control the LED stripe to implement chaseing effect while
camera.py and inference_module.py is running. This is fast developed, 
potential bugs may exist and needs reconstruct, take care.
'''

from rpi_ws281x import Adafruit_NeoPixel, Color
import time

# LED configuration
LED_COUNT = 4  # LED number
LED_PIN = 18    # GPIO
LED_FREQ_HZ = 800000
LED_DMA = 10
LED_BRIGHTNESS = 63
LED_INVERT = False
LED_CHANNEL = 0

# create NeoPixel
strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
strip.begin()

def theater_chase(color, wait_ms = 250, iterations=1):
    for j in range(iterations):
        for q in range(4):
            for i in range(0, LED_COUNT, 4):
                strip.setPixelColor(i+q, color)
            strip.show()
            time.sleep(wait_ms/1000.0)
            for i in range(0, LED_COUNT, 4):
                strip.setPixelColor(i+q, Color(0, 0, 0))

def main():
    try:
        start_time = time.time()
        while True:
            # theater_chase(Color(255, 0, 0))  # red
            theater_chase(Color(255, 255, 255))  # green
            # theater_chase(Color(0, 0, 255))  # blue
            elapsed_time = time.time() - start_time
            if elapsed_time > 18.5:
                for i in range(4):
                    strip.setPixelColor(i, Color(0, 0, 0))
                strip.show()
                break
            
    except KeyboardInterrupt:
        # rurn off leds
        for i in range(4):
            strip.setPixelColor(i, Color(0, 0, 0))
            break 
        strip.show()

if __name__ == "__main__":
    main()

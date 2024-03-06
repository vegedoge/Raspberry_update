'''
File name: main_task.py
Author: WU Yuxuan
Created time: 2024.1.25
Description: successed from Qu,Dean(GDE-EDECI)'s sensor running module, 
added other functions needed.
this file realizes all workflow on raspberry for ffd project.
including sensor data collection, data process, model inference
and bluetooth transmission.
Specially: Can be divided to 3 body modules and 3 interruption modules 
respectively: run_sensors -> Inference -> Bluetooth transmission
& Reed detect | LED enable | Buzzer beep
Further info can be found in ffd_demo_pi.md
'''

import os
import threading
import subprocess
import time
import RPi.GPIO as GPIO
import pandas as pd
import csv
from datetime import datetime
import json
from collections import deque
import shutil

# own modules
import enose
import camera
import sgp44 as sgp
import inference_module as Model
import sound as Sound
import bluetooth_module_long as BT
import get_gas_data as GetData
# import light_stripe as LightStripe
import json_decode as decode
# import chase_led as Chase_led


WORK_DIR_f = '/home/pi/ffd_demo/ffd-logging/'

class main_task(object):
    def __init__(self, event):
        
        self.event = event
        
        with open('/home/pi/Downloads/device_name.txt','r') as f :
            self.device_name=f.readline()[:-1]

        #create dir
        if not os.path.exists(WORK_DIR_f):
            os.mkdir(WORK_DIR_f)
            
        # initialize work folder, preventing time conflict
        self.init_folder("/home/pi/ffd_demo/ffd-logging")
        self.init_pic_folder("/home/pi/ffd_demo/ffd_picture")
            
        self.date = time.strftime('%Y-%m-%d',time.localtime(time.time()))
        #create child dir
        self.WORK_DIR_c = WORK_DIR_f + self.device_name + '-' +self.date + '/'
        if not os.path.exists(self.WORK_DIR_c):
            os.mkdir(self.WORK_DIR_c)
        
        # wait for open device
        time.sleep(5)
        
        # speed up camera exposture time
        subprocess.Popen(['python3','/home/pi/ffd_demo/camera_speed_up.py'])
        time.sleep(5)
        
        # initialize peripheral objects
        try:
            self.camera_instance = camera.camera()
            # self.LED_instance = LightStripe.LightStripe()
            self.sound_player = Sound.SoundPlayer()
            self.bt_instance = BT.BluetoothModule(event)
        except Exception as e:
            print(f"Device start failed exception: {e}")
            pass
        
        ''' BME688 sensor disabled
        subprocess.Popen(['chmod','777','/home/pi/ffd_demo/ejecutable'])
        #running bme688 sensor
        subprocess.Popen(['/home/pi/ffd_demo/ejecutable', (self.WORK_DIR_c+self.date+'-bme.csv')])
        time.sleep(5)
        
        self.preBmeLine = int(subprocess.check_output(['wc', '-l', (self.WORK_DIR_c+self.date+'-bme.csv')]).split()[0])

        self.rebootbme = 0
        self.timeIntervalBme = time.time()
        '''
        
        try:
            #
            self.sgp_running=0
            self.sgp_instance=sgp.sgp()
            self.sgp_running=1
        except:
            pass
        time.sleep(2)
        try :
            self.enose_running=0
            self.enose_instance = enose.eNose()
            self.enose_running=1
        except:
            pass
        time.sleep(2)
        
        #create files for save data
        self.sgp_file = self.WORK_DIR_c + self.date + '-sgp.csv'
        if not os.path.exists(self.sgp_file):
            with open(self.sgp_file, 'w') as f:
                csv_f = csv.writer(f)
                csv_f.writerow(['worldtime','Cycle','Temp1','Temp2','Temp3','Temp4','S1_raw','S2_raw','S3_raw','S4_raw'])
        
        # original enose file receives IIC message now
        self.I2C_file = self.WORK_DIR_c + self.date + '-I2C_message.csv'
        if not os.path.exists(self.I2C_file):
            with open(self.I2C_file, 'w') as f:
                csv_f = csv.writer(f)
                csv_f.writerow(['worldtime','System_State','State_Machine','Reed_Timestamp0','Reed_Timestamp1','Reed_Timestamp2','Reed_Timestamp3','Device_Timestamp0','Device_Timestamp1','Device_Timestamp2','Device_Timestamp3','GM-302BLo','GM-302BHi','GM-502BLo','GM-502BHi','MQ137Lo','MQ137Hi','crc'])
        
        # data extracting is needed, this csv file extracts the data needed for inference  
        self.enose_file = self.WORK_DIR_c + self.date + '-enose.csv'
        if not os.path.exists(self.enose_file):
            with open(self.enose_file, 'w') as f:
                csv_f = csv.writer(f)
                csv_f.writerow(['worldtime','enose01_h','enose01_l', 'enose02_h','enose02_l', 'enose03_h','enose03_l', 'crc'])

        # label 
        # self.button_file = self.WORK_DIR_c + self.date + '-label.json'
        # if not os.path.exists(self.button_file):
        #     with open(self.button_file, 'w') as f:
        #         keys = ["Measure object", "Texture", "Odor", "Perception", "Color", "Bacterial count", "Spoilage", "Freshness", "Normal"]
        #         d = {k: [] for k in keys}
        #         json.dump(d,f)  
    
    def run_local_task(self, duration):
        # use flag to control testing duration
        # 0 for closed, 1 for open
        # size can be set arbitrarily longer, longer
        reed_history = deque([0, 0], 2)
        
        elapsed_time = 0
        loop_time = 0
        while True:
            # time control
            # run loop for sleep_time sec every loop_time sec 
            # print(f"local task: loop time -> {loop_time}")
            # delete "print" for real "time"
            if list(reed_history) == [1, 0]:
                loop_time = 0
                self.bt_instance.block = 1
                try:
                    print("reed status changed, start detecting")
                    print("local task: start processing")
                    # self.camera_instance.work()
                    self.sound_player.work()
                    self.led_chase()
                    self.camera_instance.work()
                    self.run_get_data()
                    self.run_inference()
                    control_bit = decode.json_decode()
                    print(control_bit)
                    # self.LED_instance.update_status(control_bit)
                    self.sound_player.work_end()
                    self.led_process_control(control_bit)
                    self.update_bluetooth()
                    print("local task: processing finished")
                    self.clean_folder("/home/pi/ffd_demo/ffd_picture")
                    self.clean_subfolders("/home/pi/ffd_demo/ffd-logging")
                    self.bt_instance.block = 0
                except Exception as e:
                    self.bt_instance.block = 0
                    print(f'local task: proecssing failed: {e}')
                    
            # elif to make sure detection won't happen while the drawer is open
            # every 3600 sec, run inference once    
            elif loop_time > 60 and list(reed_history) == [0, 0]:
                loop_time = 0
                self.bt_instance.block = 1
                # run inference & transmission
                try:
                    print("time interval reached")
                    print("local task: start processing")
                    self.led_chase()
                    self.camera_instance.work()
                    self.run_get_data()
                    self.run_inference()
                    control_bit = decode.json_decode()
                    # self.LED_instance.update_status(control_bit)
                    self.led_process_control(control_bit)
                    self.update_bluetooth()
                    print("local task: processing finished")
                    self.clean_folder("/home/pi/ffd_demo/ffd_picture")
                    self.clean_subfolders("/home/pi/ffd_demo/ffd-logging")
                    self.bt_instance.block = 0
                except Exception as e:
                    self.bt_instance.block = 0
                    print("local task: proecssing failed")
                
                # print(f"local task: sleeping {duration} seconds")   
                # time.sleep(duration)
            # refresh if received message
            elif self.event.is_set():
                self.bt_instance.block = 1
                loop_time = 0
                try:
                    self.event.clear()
                    print("refresh message reveived, start detecting")
                    print("local task: start processing")
                    # self.camera_instance.work()
                    self.sound_player.work()
                    self.led_chase()
                    self.camera_instance.work()
                    self.run_get_data()
                    self.run_inference()
                    control_bit = decode.json_decode()
                    print(control_bit)
                    # self.LED_instance.update_status(control_bit)
                    self.sound_player.work_end()
                    self.led_process_control(control_bit)
                    self.update_bluetooth()
                    print("local task: processing finished")
                    self.clean_folder("/home/pi/ffd_demo/ffd_picture")
                    self.clean_subfolders("/home/pi/ffd_demo/ffd-logging")
                    self.bt_instance.block = 0
                except Exception as e:
                    self.bt_instance.block = 0

                    print(f'local task: refresh proecssing failed: {e}')
                
            start_time = time.time()
            loop_time = loop_time + elapsed_time
            
            # sgp write to csv
            if self.sgp_running==1:
                try:
                    mesg_sgp=self.sgp_instance.getcontent()
                    if mesg_sgp != None:
                        # print(mesg_sgp)
                        with open (self.sgp_file, 'r',errors='ignore') as f :
                            # control file size
                            reader = csv.reader(f)
                            num_rows = sum(1 for row in reader)
                            
                        # 2000 rows for sgp
                        max_rows = 2000
                        # clean size for one time
                        del_interval = 1000
                        
                        # delete older rows if exceeding
                        if num_rows > max_rows:
                            self.delete_rows(self.sgp_file, del_interval)
                            
                        # write in new lines
                        with open (self.sgp_file, 'a') as f : 
                            csv_f = csv.writer(f)
                            csv_f.writerow([datetime.fromtimestamp(float(mesg_sgp[0])).strftime("%Y-%m-%d %H:%M:%S.%f "),mesg_sgp[1],mesg_sgp[2],mesg_sgp[3],mesg_sgp[4],mesg_sgp[5],mesg_sgp[6],mesg_sgp[7],mesg_sgp[8],mesg_sgp[9]])
                except Exception as e:
                    print(f"sgp csv bug: {e}")
                    with open(self.sgp_file, 'r') as f:
                        lines = f.readlines()
                    with open(self.sgp_file, 'w') as f:
                        f.writelines(lines[:-1])
                
            # enose I2C write to csv
            if self.enose_running==1:
                try:
                    mesg_enose=self.enose_instance.getcontent()
                    if mesg_enose != None:
                        print(mesg_enose)
                        print(f"enose_time: {float(mesg_enose[0])}")
                        
                        # reed status detection
                        if int(mesg_enose[1]) == 16:# 00010000
                            reed_status = 1
                        elif int(mesg_enose[1]) == 24:# 00011000
                            reed_status = 0
                        else:
                            reed_status = 9
                        
                        if reed_status == 0:
                            reed_str = "closed"
                        else:
                            
                            reed_str = "open" 
                        print(f"reed status:{reed_str}" )
                        reed_history.append(reed_status)
                        
                        # 8 bit data segment, power(2, 8) is needed
                        reed_time = 256 * 256 * 256 * float(mesg_enose[6]) + 256 * 256 * float(mesg_enose[5]) + 256 * float(mesg_enose[4]) + float(mesg_enose[3])
                        reed_time = datetime.fromtimestamp(reed_time).strftime("%Y-%m-%d %H:%M:%S.%f ")
                        print(f"reed_last_change:{reed_time}")
                        
                        # file size control
                        with open (self.I2C_file, 'r', errors='ignore') as f :
                            reader = csv.reader(f)
                            num_rows = sum(1 for row in reader)
                            
                        # sgp and enose row ratio approx -> 40 : 1000
                        max_rows = 80
                        # clean size for one time
                        del_interval = 40
                        
                        # delete older rows if exceeding
                        if num_rows > max_rows:
                            self.delete_rows(self.I2C_file, del_interval)
                        
                        with open (self.I2C_file, 'a') as f :
                            csv_f = csv.writer(f)
                            csv_f.writerow([datetime.fromtimestamp(float(mesg_enose[0])).strftime("%Y-%m-%d %H:%M:%S.%f "),mesg_enose[1],mesg_enose[2],mesg_enose[3],mesg_enose[4],mesg_enose[5],mesg_enose[6],mesg_enose[7],mesg_enose[8],mesg_enose[9],mesg_enose[10],mesg_enose[11],mesg_enose[12],mesg_enose[13],mesg_enose[14],mesg_enose[15],mesg_enose[16],mesg_enose[17]])
                except Exception as e:
                    # fix nul bug
                    print(f"enose csv bug: {e}")
                    with open(self.I2C_file, 'r') as f:
                        lines = f.readlines()
                    with open(self.I2C_file, 'w') as f:
                        f.writelines(lines[:-1])
            #for check bme running
            ''' BME688 sensor disabled
            if time.time() - self.timeIntervalBme > 60 and self.rebootbme == 0 :
                aftBmeLine = int(subprocess.check_output(['wc', '-l', (self.WORK_DIR_c+self.date+'-bme.csv')]).split()[0])
                if aftBmeLine == self.preBmeLine :
                    print('no bme data for 1 mins')
                    subprocess.Popen(['reboot'])
                    time.sleep(10)
                    self.rebootbme = 1
                else:
                    self.preBmeLine = aftBmeLine
                    self.timeIntervalBme = time.time()
            '''

            elapsed_time = time.time() - start_time
            
                  
    def run_get_data(self):
        '''
        data preprocess for later inference op 
        '''
        try : 
            print("data preprocess start")
            self.pre_data = GetData.GetData()
            print("data preprocess finished")
        except :
            print("data preprocess failed")
            pass
                    
    def run_inference(self):
        '''
        start inference module and write json file to 
        target folder
        ''' 
        try : 
            print("Inference Model initialized")
            self.model = Model.Inference_Module()
            print("Inference finished")
        except :
            print("Inference failed")
            pass
    
    def update_bluetooth(self):
        '''
        update bluetooth data
        '''
        try :
            # json data to be sent --test
            print("bluetooth update and connecting")
            
            # read json file and load as dic
            
            # data_to_send_file = "result_data.json"
            data_to_send_file = "/home/pi/ffd_demo/result_data.json"
            
            with open(data_to_send_file, 'r') as json_file:
                data_to_send = json.load(json_file)  
            print(data_to_send)
            self.bt_instance.update_data(data_to_send)
            print("bluetooth data updated")
        except Exception as e:
            print(f"Bluetooth Failed exception: {e}")
            pass
        
    def clean_folder(self, folder_path, max_files=10):
        '''control the size of ffd picture file'''
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
        num_files = len(files)
        if num_files <= max_files:
            return
        num_files_to_delete = num_files - max_files
        files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
        
        # iterate target files
        for i in range(num_files_to_delete):
            os.remove(os.path.join(folder_path, files[i]))
            print(f"Deleted old file: {files[i]}")
    
    def clean_subfolders(self, folder_path, max_subfolders=5):
        '''control the size of ffd-logging file size'''
        subfolders = [subfolder for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]
        
        num_subfolders = len(subfolders)
        if num_subfolders <= max_subfolders:
            return
        num_subfolders_to_delete = num_subfolders - max_subfolders
        subfolders = sorted(subfolders, key=lambda subfolder: os.path.getmtime(os.path.join(folder_path, subfolder)))
        
        # iterate subfolders to be deleted
        for i in range(num_subfolders_to_delete):
            # delete olderst folders
            subfolder_path = os.path.join(folder_path, subfolders[i])
            for root, dirs, files in os.walk(subfolder_path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                os.rmdir(root)
            print(f"Deleted old subfolder: {subfolders[i]}")
    
    def delete_rows(self, csv_file, num_rows_to_delete = 1000):
        '''
        delete num_rows from start of csv file
        not including the title row(1st)
        '''
        with open(csv_file, 'r') as file:
            lines = file.readlines()
        with open(csv_file, 'w') as file:
            file.writelines(lines[:1] + lines[num_rows_to_delete + 1:])
       
    def init_folder(self, folder_path):
        '''
        Every time when booted, clear all old data files
        in case time module goes back to certain point and cause
        mismatch of data and time
        '''
        # get all contents of one folder
        files = os.listdir(folder_path)
        # clear all subfolers and anything inside
        for file in files:
            file_path = os.path.join(folder_path, file)
            shutil.rmtree(file_path)
            
    def init_pic_folder(self, folder_path):
        '''
        Every time when booted, clear all old pic files
        in case time module goes back to certain point and cause
        mismatch of data and time
        '''
        # get all contents of one folder
        files = os.listdir(folder_path)
        # clear all pic files inside
        for file in files:
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
        
    def led_process_control(self, control_bit = "0000"):
        '''multiprocess initialization for sudo LED control'''
        subprocess.run(["sudo", "python3", "/home/pi/ffd_demo/light_stripe_single.py", "--led_status", control_bit])

    def led_chase(self):
        '''run chasing led while camera and inference under processing'''
        subprocess.Popen(["sudo", "python3", "/home/pi/ffd_demo/chase_led.py"])

        
                
if __name__ =='__main__':
    refresh_event = threading.Event()
    app = main_task(refresh_event)
    # 1 min for each time
    duration = 30
    app.run_local_task(duration)

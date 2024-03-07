'''
File name: inference_module.py
Edited by: WU Yuxuan
Edit time: 2024.1.25
Description: implement cv and gas model inference in this module,
pictures are divided into 4 * 4 grids and deployed multitask model
separately. Then gas module inference is deployed. finally we write 
a json file containing indexes as our BT communication message. 
'''

import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
# from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
import pandas as pd
import json
import time
import cv2
import pdb

class Inference_Module:
    '''
    Implement inference for cv model & gas model
    input image path and csv path
    output json file to queue
    '''
    def __init__(self):
        
        print("inference initialized")
        self.run_inference()


    def split_image_into_grid(self, image, rows=4, cols=4):
        width, height = image.size
        grid_width = width // cols
        grid_height = height // rows
        grid_images = []
        
        # path for saving divided images
        divided_image_path = "/home/pi/ffd_demo/divided_images"

        for i in range(cols):  # Iterate over columns first
            col_images = []
            for j in range(rows):  # Then iterate over rows
                left = i * grid_width
                upper = j * grid_height
                right = (i + 1) * grid_width
                lower = (j + 1) * grid_height

                grid_image = image.crop((left, upper, right, lower))
                col_images.append(grid_image)
                
                # save sub image
                # save sub image file path
                sub_image_name = "divided image-col" + str(i+1) + '-row' + str(j+1) + '.jpg'
                sub_image_path = os.path.join(divided_image_path, sub_image_name)

                # save to path
                grid_image.save(sub_image_path)

            grid_images.append(col_images)

        return grid_images

    def class_to_name(self, classes,detection_type):
        category2name = {0:"Background", 1:"Blueberry", 2:"Beef"}
        class2fresh = {0:"Fresh", 1:"Spoiled"}
        class2fresh_three = {0:"Fresh", 1:"Semi Fresh", 2:"Spoiled"}
        if detection_type.lower()=="blueberry" or detection_type.lower()=="beef":
            temp_class2fresh = class2fresh if classes == 2 else class2fresh_three
            return temp_class2fresh
        else:
            return category2name
        
    def get_all_samples_from_one_folder(self, folder):
            supported = ["jpg", "JPG", "png", "PNG"]  # Supported file extensions
            image_paths = [os.path.join(folder, i) for i in os.listdir(folder)
                        if i.split('.')[-1] in supported] 
            return sorted(image_paths)

    def load_and_run_tflite_model_cv(self, model_path, image):
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get the indices of input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Check the shape and data type of the input tensor
        input_shape = input_details[0]['shape']
        input_data_type = input_details[0]['dtype']

        # Check the shape and data type of the output tensor
        output_shape = output_details[0]['shape']
        output_data_type = output_details[0]['dtype']

        # Load and preprocess the image
        # image = Image.open(image_path)
        
        image = image.resize((input_shape[1], input_shape[2]),Image.BICUBIC)
        image = np.array(image, dtype=np.float32)
        print(image.shape, '2')
        
        # image = np.array(image, dtype=np.uint8) ##teshu shiyong
        
        # original

        # image /= 255.0  # Normalize pixel values  
        # image -= [0.485,0.456,0.406]
        # image /= [0.229,0.224,0.225]



        
        def normalize_cv2(img: np.ndarray, mean: np.ndarray, denominator: np.ndarray) -> np.ndarray:
            if mean.shape and len(mean) != 4 and mean.shape != img.shape:
                mean = np.array(mean.tolist() + [0] * (4 - len(mean)), dtype=np.float64)
            if not denominator.shape:
                denominator = np.array([denominator.tolist()] * 4, dtype=np.float64)
            elif len(denominator) != 4 and denominator.shape != img.shape:
                denominator = np.array(denominator.tolist() + [1] * (4 - len(denominator)), dtype=np.float64)
        
            img = np.ascontiguousarray(img.astype("float32"))
            cv2.subtract(img, mean.astype(np.float64), img)
            cv2.multiply(img, denominator.astype(np.float64), img)
            return img
        
        def normalize(img: np.ndarray, mean: np.ndarray, std: np.ndarray, max_pixel_value: float = 255.0) -> np.ndarray:
            mean = np.array(mean, dtype=np.float32)
            mean *= max_pixel_value
        
            std = np.array(std, dtype=np.float32)
            std *= max_pixel_value
        
            denominator = np.reciprocal(std, dtype=np.float32)
        
            # if img.ndim == 3 and img.shape[-1] == 3:
            return normalize_cv2(img, mean, denominator)
            # return normalize_numpy(img, mean, denominator)
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = normalize(image, mean, std)
        
        
        
        
        # pdb.set_trace()
        
        #
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        
        # image = (image - mean)/std
        # image = image.astype(np.float32)
        
        # index = 0
        # while index < 5:
        #     print(f'image first:{image[index]}')
        #     index = index + 1

        # image -= [0.5,0.5,0.5]
        # image /= [0.5,0.5,0.5]
        image = np.expand_dims(image, axis=0)

        # Pass the image data to the input tensor
        interpreter.set_tensor(input_details[0]['index'], image)
        
        
        
        

        # Run the model
        interpreter.invoke()

        # Get the output of the model
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        output_len = len(output)

        # Perform post-processing on the output
        print(f"output:{output}")
        print(f"size of output:{output_len}")
        # class_probabilities = np.exp(output) / np.sum(np.exp(output), axis=0)

        # if using packaged-model, returns normal info
        # if using unpackaged-model, package_info is meangingless
        if output_len == 11:
            package_info = output[0:2]
            grouped_output = np.split(output[2:], 3)
        elif output_len == 9:
            package_info = output[0:2]
            grouped_output = np.split(output, 3)
            
        class_probabilities = [np.exp(group) / np.sum(np.exp(group)) for group in grouped_output]
        class_predicted = [np.argmax(class_probability) for class_probability in class_probabilities]

        return package_info, class_predicted,class_probabilities

    def write_to_image(self, 
                    args,
                    category_index,
                    category_probability_list,
                    beef_index,
                    beef_probability_list,
                    blueberry_index,
                    blueberry_probability_list,
                    original_image_dir
                    ):
        import cv2
        category_class=self.class_to_name(args.class_number,"category")[category_index]
        category_probablity=category_probability_list[category_index]
        beef_class=self.class_to_name(args.class_number,"beef")[beef_index]
        beef_probability=beef_probability_list[beef_index]
        blueberry_class=self.class_to_name(args.class_number,"blueberry")[blueberry_index]
        blueberry_probability=blueberry_probability_list[blueberry_index]
        label_category = f'Category:{category_class}, probablity:{category_probablity}'
        label_beef = f'beef:{beef_class}, probablity:{beef_probability}'
        label_blueberry = f'blueberry:{blueberry_class},probablity:{blueberry_probability}'
        image=cv2.imread(original_image_dir)
        cv2.putText(image, label_category, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, label_beef, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, label_blueberry, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        sub_path=""
        if args.quantize.lower()=="disabled":
            sub_path="quantize_disabled"
        else:
            sub_path=args.quantize.lower()

        if not os.path.exists(os.path.join(os.path.split(args.output_image_path)[0])):
            os.mkdir(os.path.join(os.path.split(args.output_image_path)[0]))
        if not os.path.exists(os.path.join(os.path.split(args.output_image_path)[0],sub_path)):
            os.mkdir(os.path.join(os.path.split(args.output_image_path)[0],sub_path))
        cv2.imwrite(os.path.join(os.path.split(args.output_image_path)[0],sub_path, os.path.splitext(original_image_dir)[0].split("/")[-1]+"_predicted"+".jpg"), image)

    def load_and_run_tflite_model_en(self, model_path, data_path, inference_type):
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get the indices of input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Check the shape and data type of the input tensor
        input_shape = input_details[0]['shape']
        input_data_type = input_details[0]['dtype']

        # Check the shape and data type of the output tensor
        output_shape = output_details[0]['shape']
        output_data_type = output_details[0]['dtype']

        # Load and preprocess the csv file
        test_data = pd.read_csv(data_path)
        X_test = test_data.drop(['worldtime', 'state', 'index', 'crc'], axis=1)
        # y_test = test_data['spoliage']

        # Pass the csv data to the input tensor
        if inference_type.lower()=="single":
            # render last row as input
            input_X=X_test.values[-1].astype(np.float32)
            input_X=np.expand_dims(input_X, axis=0)
            interpreter.set_tensor(input_details[0]['index'], input_X)
            inpute_data = input_X.dtype
            input_data_type = input_details[0]['dtype']
        else:
            interpreter.set_tensor(input_details[0]['index'], X_test.values.astype(np.float32))

        # Run the model
        interpreter.invoke()

        # Get the output of the model
        predictions = interpreter.get_tensor(output_details[0]['index'])
        confidence=interpreter.get_tensor(output_details[1]['index'])
        
        # if inference_type.lower()=="batch":
        #     accuracy = accuracy_score(y_test, predictions)
        #     confusion_mat = confusion_matrix(y_test, predictions)
        # else:
        #     accuracy=None
        #     confusion_mat=None

        return predictions,confidence # ,accuracy,confusion_mat

    def get_freshness_based_on_category(self, category_index, beef_index, blueberry_index):
        if category_index == 0:
            freshness = -1
        elif category_index == 1:
            freshness = blueberry_index
        elif category_index == 2:
            freshness = beef_index
        else:
            # 如果category_index不是0, 1, 2中的任何一个值，可以根据实际情况处理，这里假设输出为-1
            freshness = -1
        
        return freshness
    
    def run_inference(self):
        # inference model initialization
        
        # args initialization 
        # cv args
        model_path_cv = "/home/pi/ffd_demo/CV_conversion/EfficientNet_V2S/tflite_out_3classes/"
        output_image_path = "/home/pi/ffd_demo/test_pictures"
        class_number = 3
        # quantize = "disabled"
        quantize = "int8"
        write_to_image = "enabled"
        
        # select picture by time
        picture_folder_path = "/home/pi/ffd_demo/ffd_picture"
        # pictures = [f for f in os.listdir(picture_folder_path) 
        #                 if os.path.isdir(os.path.join(picture_folder_path, f))]
        
        pictures = [f for f in os.listdir(picture_folder_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # choose latest subfolder path
        # sort by time
        pictures.sort(key=lambda f: os.path.getmtime(os.path.join(picture_folder_path, f)))
        latest_picture = pictures[-1]
        latest_picture_path = os.path.join(picture_folder_path, latest_picture)
        # path reload
        image_path = latest_picture_path
        
        # test path
        # image_path = "/home/pi/ffd_demo/demo_cv_test/BSH/abc2024-02-29-11-35-06.jpg"
        
        # test original pic
        # image_path = "./lab_beef.jpg"
        # pic enhance test path
        # image_path = "./enhanced.jpg"
        
        
        # gas args
        model_path = "/home/pi/ffd_demo/ML_conversion/XGBoost/tflite_out_xgboost_single/"
        model_file = "xgboost_separated_by_box_test_on_box1.tflite"
        csv_path = "/home/pi/ffd_demo/ffd_logging_cache/sgp_enose.csv"
        inference_type = "single"
        
        # create an empty for data storage
        data_list = []
        
        #      ------- get_cv_information -------
        # Assuming 'image' is the original image you loaded
        # image_list = self.get_all_samples_from_one_folder(image_path)
        sub_path = ""

        if quantize.lower() == "disabled":
            sub_path = "quantize_disabled"
        else:
            sub_path = quantize.lower()

        # image_path = image_list[1]
        original_image = Image.open(image_path)

        # Split the image into three columns
        grid_images = self.split_image_into_grid(original_image, rows=4, cols=4)
        print(f'original image:{original_image.size}')
        print(original_image.mode)
        
        column_data_list = []
        # Loop through each column and process it with the model
        for i, col_images in enumerate(grid_images):
            current_column_data = []
            for j, grid_image in enumerate(col_images):
                # Assuming 'grid_image' is the cropped grid image

                # quantize disabled
                
                
                
                # unpackaged model int8
                # package_info, result_index, result_possibilities = self.load_and_run_tflite_model_cv(
                #     os.path.join(model_path_cv, sub_path, "combined_model_dict_v0.2.1.tflite"), grid_image
                # )
                
                # packaged model int8
                package_info, result_index, result_possibilities = self.load_and_run_tflite_model_cv(
                    os.path.join(model_path_cv, sub_path, "demo_model_240229.tflite"), grid_image
                )
                

                # Do something with the results for each grid image
                category_index = result_index[0]
                category_possibilities = result_possibilities[0]
                # credential check
                if category_possibilities[category_index] < 0.9:
                    category_index = 0
                blueberry_index = result_index[1]
                blueberry_possibilities = result_possibilities[1]
                beef_index=result_index[2]
                beef_possibilities = result_possibilities[2]

                freshness_result = self.get_freshness_based_on_category(category_index, beef_index, blueberry_index)
                print("Freshness:", freshness_result,"category_index",category_index)
                current_column_data.append({"freshness": freshness_result, "category_index": category_index})

                
                print(f"Results for grid {i + 1}-{j + 1}:")
                print(f"Predicted category: {category_index}, probabilities: {category_possibilities}")
                print(f"Predicted blueberry freshness: {blueberry_index}, probabilities: {blueberry_possibilities}")
                print(f"Predicted beef freshness: {beef_index}, probabilities: {beef_possibilities}")
                print("\n")
                
            column_data_list.append(current_column_data)
            predict,confidence = self.load_and_run_tflite_model_en(
                os.path.join(model_path,sub_path, model_file), csv_path, inference_type)
        
        print(f"predicted from en: {predict}")

        for col_index, column_data in enumerate(column_data_list):
            # initialize related index
            blueberry_exist = 0
            beef_exist = 0
            cv_worst_blueberry = -1
            cv_worst_beef = -1

            # check category_index for every data segment
            for data_point in column_data:
                category_index = data_point["category_index"]

                # update index according to flag 1 or 2
                if category_index == 1:
                    blueberry_exist = 1
                elif category_index == 2:
                    beef_exist = 1

                # update max freshness
                if category_index == 1 and data_point["freshness"] > cv_worst_blueberry:
                    cv_worst_blueberry = data_point["freshness"]
                if category_index == 2 and data_point["freshness"] > cv_worst_beef:
                    cv_worst_beef = data_point["freshness"]

            # create data dic and add to data_list
            temp_enose = 0
            # test case temp_enose
            predict = temp_enose
            
            data = {
                "column": col_index + 1,
                "blueberry_exist": int(blueberry_exist),
                "beef_exist": int(beef_exist),
                "cv_worst_blueberry": int(cv_worst_blueberry),
                "cv_worst_beef": int(cv_worst_beef),
                "EN Freshness": int(predict)
            }
            data_list.append(data)

        # write the whole list to json file
        json_filename = '/home/pi/ffd_demo/result_data.json'
        with open(json_filename, 'w') as json_file:
            json.dump(data_list, json_file)

if __name__ == "__main__":
    test = Inference_Module()
    while True:
        print("inference module ongoing")
        time.sleep(5)
    
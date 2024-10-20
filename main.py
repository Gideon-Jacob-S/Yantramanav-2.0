import serial
import cv2
from ultralytics import YOLO
import numpy as np
from keras.models import load_model 
import asyncio, time, os

HOME = os.getcwd()
food_package_model = YOLO(fr"{HOME}\models\food_or_package_detection.pt")
logo_segment_model = YOLO(fr"{HOME}\models\logo_and_details_detection.pt")
brand_classify_model = YOLO(fr"{HOME}\models\brand_classification.pt")
food_classify_model = YOLO(fr"{HOME}\models\food_classification.pt")
food_freshness_model = load_model(fr"{HOME}\models\food_freshness.h5")

arduino = serial.Serial(port='COM4', baudrate=9600, timeout=1)
reject_item_queue = []
main_result = {}

package_info = {0: [], 1: [], 2: []}
instance_result = None 

camera_ports = [0,2]
images = []

def read_camera_input():
    for port in camera_ports:
        cam = cv2.VideoCapture(port)
        result, image = cam.read() 
        images.append(image)

    # images.append(cv2.imread(fr"{HOME}\samples\food1.jpg", cv2.IMREAD_COLOR))
    # images.append(cv2.imread(fr"{HOME}\samples\food2.jpg", cv2.IMREAD_COLOR))

    # images.append(cv2.imread(fr"{HOME}\samples\package1.jpg", cv2.IMREAD_COLOR))
    # images.append(cv2.imread(fr"{HOME}\samples\package2.jpg", cv2.IMREAD_COLOR))

def show_images(images=images):
    for img in images:
        cv2.imshow("Image", img)
        cv2.waitKey(0) 
        cv2.destroyWindow("Image") 

def crop_image(coordinates, image):
    x1, y1, x2, y2 = coordinates 
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

def pre_proc_img(img):
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess the image
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_processed_results(result1, result2): 

    """
    returns food's and product's count and it's cropped images
    """

    objects_count1 = {
        0: {
            'count':0,
            'images': []
        },
        1: {
            'count': 0,
            'images': []
        }
    }
    objects_count2 = {
        0: {
            'count':0,
            'images': []
        },
        1: {
            'count': 0,
            'images': []
        }
    }

    def get_cls_name(args):
        id, count = args
        return (result1[0].names[id], count)
    
    def list_to_int(my_list):
        return list(map(int, my_list))[:4]


    result1_cls_data = map(int, result1[0].boxes.cls)
    result1_box_data = map(list_to_int, result1[0].boxes.data)

    result2_cls_data = map(int, result2[0].boxes.cls)
    result2_box_data = map(list_to_int, result2[0].boxes.data)


    for id, box in zip(result1_cls_data, result1_box_data):
        objects_count1[id]['count'] +=  1
        objects_count1[id]['images'].append(crop_image(box, images[0]))
    
    for id, box in zip(result2_cls_data, result2_box_data):
        objects_count2[id]['count'] +=  1
        objects_count1[id]['images'].append(crop_image(box, images[1]))


    objects_count1 = dict(map(get_cls_name, objects_count1.items()))
    objects_count2 = dict(map(get_cls_name, objects_count2.items()))
    
    objects_count = {
        'food': {
            'count': max(objects_count1['food']['count'], objects_count2['food']['count']),
            'images': objects_count1['food']['images'] + objects_count2['food']['images'],
            'details': {'status':""}
        },
        'package': {
            'count': max(objects_count1['package']['count'], objects_count2['package']['count']),
            'images': objects_count1['package']['images'] + objects_count2['package']['images'],
            'details': {'status':""}
        }
    }

    return objects_count

async def predict_model(model, img):
    return model.predict(img, device='cuda', save=False)

async def async_predict_model(model, img):
    return await predict_model(model, img)

async def predict_food_name_and_freshness(result_obj):
    for img in result_obj['images']:
        
        res = await asyncio.create_task(async_predict_model(food_classify_model, img))
        res2 = await asyncio.create_task(async_predict_model(food_freshness_model, pre_proc_img(img)))

        class_id = res[0].probs.top5[0]
        class_id_dict = res[0].names
        class_name = class_id_dict[class_id]

        if class_name not in result_obj['details']:
            result_obj['details'][class_name] = {'count': 0, 'freshness': []}
        
        result_obj['details'][class_name]['count'] += 1
        result_obj['details'][class_name]['freshness'].append(100 - res2[0][0])
    
    fresh, not_fresh = 0, 0
    for score in result_obj['details'][class_name]['freshness']:
        if score > 60:
            fresh += 1
        else:
            not_fresh += 1
    
    for food in result_obj['details'].keys():
        if food=='status':
            continue
        freshness_array = result_obj['details'][food]['freshness']
        result_obj['details'][food]['freshness'] = round(sum(freshness_array) / len(freshness_array), 2)
    
    if not_fresh > fresh:
        result_obj['details']['status'] = "Rejected"
        return False 
    
    result_obj['details']['status'] = "Good"
    return True 

async def segement_images(img):
    global package_info, instance_result
    
    def list_to_int(my_list):
        return list(map(int, my_list))[:4]

    res = await predict_model(logo_segment_model, cv2.resize(img, (640, 640)))
    instance_result = res 
    result_cls_data = map(int, res[0].boxes.cls)
    result_box_data = map(list_to_int, res[0].boxes.data)

    for id, box in zip(result_cls_data, result_box_data):
        if id not in package_info:
            package_info[id] = []
        package_info[id].append(crop_image(box, img))

async def predict_brand_name(img_list, result_obj):
    img = img_list[0]
    res = await predict_model(brand_classify_model, cv2.resize(img, (640, 640)))

    class_id = res[0].probs.top5[0]
    class_id_dict = res[0].names
    class_name = class_id_dict[class_id]

    result_obj['details']['name'] = class_name
    
async def predict_serial_info(img_list, result_obj):
    pass

async def predict_expeiry_info(img_list, result_obj):
    result_obj['details']['status'] = "Good"

async def predict_logo_and_details(result_obj):
    global package_info, instance_result
    result_obj['details']['count'] = result_obj['count']

    def get_cls_name(args):
        id, array = args
        return (instance_result[0].names[id], array)
    
    tasks = []
    for img in result_obj['images']:
        tasks.append(asyncio.create_task(segement_images(img)))
    
    for task in tasks:
        await task 

    package_info = dict(map(get_cls_name, package_info.items()))
    
    task_object = asyncio.gather(
        predict_brand_name(package_info['logo'], result_obj),
        predict_serial_info(package_info['serial'], result_obj),
        predict_expeiry_info(package_info['details'], result_obj)
    )

    await task_object

    if result_obj['details']['status'] == "Rejected":
        return False 
    return True 

async def process_the_image():
    global main_result

    read_camera_input()
    start_time = time.time()
    ###################################################

    food_package_task = asyncio.gather(async_predict_model(food_package_model, images[0]), async_predict_model(food_package_model, images[1]))
    food_package_result1, food_package_result2 = await food_package_task

    result = get_processed_results(food_package_result1, food_package_result2)

    if result['food']['count'] > 0:
        is_good_product = await asyncio.create_task(predict_food_name_and_freshness(result['food']))
        print(f"\n{result['food']['details']}\n")
    elif result['package']['count'] > 0:
        is_good_product = await asyncio.create_task(predict_logo_and_details(result['package']))
        print(f"\n{result['package']['details']}\n")
    
    reject_item_queue.append(not is_good_product)

    ####################################################
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")


def arduino_reject_product():
    if reject_item_queue[0] is True:
        data_to_send = b"Reject\n"
        arduino.write(data_to_send)
        time.sleep(0.2)
        
        response = arduino.readline().decode('utf-8').strip()
        print(f"{response} is the data, Arduino recieved")

    reject_item_queue.pop(0)

def arduino_object_detection_trigger():
    data = arduino.readline().decode('utf-8').strip()

    # Starting the process
    ########################################################
    if "TRIGGERED" in data:
        print(data, "--> printing for Debugging")
        asyncio.run(process_the_image())
    ########################################################

    arduino_reject_product()
    data_to_send = b"Start\n"
    arduino.write(data_to_send)
    time.sleep(0.2)
    
    response = arduino.readline().decode('utf-8').strip()
    print(f"{response} is the data, Arduino recieved")

if __name__ == "__main__":
    while True:
        arduino_object_detection_trigger()
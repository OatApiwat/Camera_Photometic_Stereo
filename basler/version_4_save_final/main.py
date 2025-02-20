from Module_open_camera import OpenBaslerCamera
import cv2
import os
import time
import numpy as np
import torch
from ultralytics import YOLO
import pymssql
from datetime import datetime

folder_path = r"/home/mic/Camera_project/basler/version_4_save_final/result_image"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
window_width = 772  # ความกว้างของหน้าต่าง
window_height = 516  # ความสูงของหน้าต่าง
cv2.namedWindow('original_img',cv2.WINDOW_NORMAL)
cv2.resizeWindow('original_img', window_width, window_height)  # ปรับขนาดหน้าต่าง
cv2.namedWindow('result_img',cv2.WINDOW_NORMAL)
cv2.resizeWindow('result_img', window_width, window_height)  # ปรับขนาดหน้าต่าง
cv2.namedWindow('roi_1',cv2.WINDOW_NORMAL)
cv2.resizeWindow('roi_1', 340, 340)  # ปรับขนาดหน้าต่าง
cv2.namedWindow('roi_2',cv2.WINDOW_NORMAL)
cv2.resizeWindow('roi_2', 340, 340)  # ปรับขนาดหน้าต่าง


circles = [
    (380, 1370, 170),  # วงกลมที่ 1
    (875, 500, 170),   # วงกลมที่ 2
]

def process_2_photometric_stereo(gray_image):
    direction = 0.75
    light_directions = np.array([
    [-direction,  direction, 1],  # image_2: แสงมาจากมุมบนซ้าย
    [-direction, -direction, 1],  # image_1: แสงมาจากมุมล่างซ้าย
    [ direction, -direction, 1],  # image_4: แสงมาจากมุมล่างขวา
    [ direction,  direction, 1],  # image_3: แสงมาจากมุมบนขวา
    ], dtype=np.float32)
    # แปลงภาพทั้งหมดเป็น numpy array
    np_images = np.stack(gray_image, axis=-1).astype(np.float32)
    # Normalization ของทิศทางแสง
    light_directions = light_directions
    light_directions /= np.linalg.norm(light_directions, axis=1, keepdims=True)
    # คำนวณ Normal Map
    height, width, num_images = np_images.shape
    intensity = np_images.reshape(-1, num_images)
    normal_map = np.dot(np.linalg.pinv(light_directions), intensity.T).T
    normal_map = normal_map.reshape(height, width, 3)

    # Normalize เพื่อให้ normal map เป็น [0, 1]
    normal_map_magnitude = np.linalg.norm(normal_map, axis=2, keepdims=True)
    normal_map /= np.maximum(normal_map_magnitude, 1e-5)
    normal_map_visual = ((normal_map + 1) / 2 * 255).astype(np.uint8)  # สำหรับแสดงผล
    
    (B, G, R) = cv2.split(normal_map_visual)
    B = cv2.medianBlur(B, 5)  # ใช้ขนาด kernel 5x5

    return normal_map_visual,B,G,R

def process_3_crop_ROI(B_image):
    #offset = 65

    crop_image = []
    for i, (x, y, r) in enumerate(circles, start=1):
        mask = np.zeros_like(B_image, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        cropped_image = cv2.bitwise_and(B_image, B_image, mask=mask)
        x1, y1 = max(0, x - r), max(0, y - r)
        x2, y2 = min(B_image.shape[1], x + r), min(B_image.shape[0], y + r)
        cropped_circle = cropped_image[y1:y2, x1:x2]
        # สร้าง mask ของวงกลมในภาพย่อยเพื่อให้ภาพมีขอบโปร่งใส
        circle_mask = np.zeros((2*r, 2*r), dtype=np.uint8)
        cv2.circle(circle_mask, (r, r), r, 255, -1)

        # ใช้ mask กับภาพที่ครอบ
        h, w = cropped_circle.shape[:2]
        final_mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = min(r, w // 2), min(r, h // 2)
        cv2.circle(final_mask, (cx, cy), r, 255, -1)

        circle_only = cv2.bitwise_and(cropped_circle, cropped_circle, mask=final_mask)
        
        # คำนวณสีเฉลี่ยจากบริเวณขอบของวงกลม
        border_mask = np.zeros_like(final_mask)
        cv2.circle(border_mask, (cx, cy), r, 255, thickness=1)
        border_pixels = cropped_circle[border_mask == 255]  # พิกเซลในบริเวณขอบ

        if len(border_pixels) > 0:  # ตรวจสอบว่าพบพิกเซล
            average_color = border_pixels.mean(axis=0).astype(np.uint8)  # ค่าสีเฉลี่ย (BGR)
            circle_only[final_mask == 0] = average_color  # เติมสีเฉลี่ยในบริเวณที่ไม่มีข้อมูล
        crop_image.append(circle_only)
    
    return crop_image

def process_4_image_classification(roi_images):
    predict_image = []
    text_result = []
    for roi_image in roi_images:
        results = model(roi_image,save = False, verbose=False)
        if results[0].probs is not None:
            probs = results[0].probs.data.cpu().numpy()  # แปลง Tensor เป็น NumPy array
            # ตรวจสอบผลลัพธ์ความมั่นใจและตัดสิน class
            conf_result = []
            class_result = []
            for cls_id, conf in enumerate(probs):
                class_name = results[0].names[cls_id]  # แปลง id เป็นชื่อคลาส
                conf_result.append(conf)
                class_result.append(class_name)
            # กำหนดผลลัพธ์ final_result ตามเงื่อนไขที่ให้มา
            if conf_result[0] > conf_result[1] and conf_result[0] > 0.6:
                final_result = class_result[0]
                circle_color = (0, 0, 255)  # สีแดง
                text_color = (0, 0, 255)    # สีแดง
            elif conf_result[1] > conf_result[0] and conf_result[1] > 0.6:
                final_result = class_result[1]
                circle_color = (0, 255, 0)  # สีเขียว
                text_color = (0, 255, 0)    # สีเขียว
            else:
                final_result = 'undefined'
                circle_color = (255, 0, 0)  # สีน้ำเงิน
                text_color = (255, 0, 0)    # สีน้ำเงิน
            if len(roi_image.shape) == 2 or roi_image.shape[2] == 1:
                roi_image = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
            # เพิ่มข้อความผลลัพธ์ในภาพ
            text = f"Result: {final_result}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(roi_image, text, (10, 30), font, 1, text_color, 2, cv2.LINE_AA)
            predict_image.append(roi_image)
            text_result.append(final_result)
    
    return predict_image,text_result

def process_5_draw_result(original_image, text_result):
    # วาดวงกลมใน original_image
    if len(original_image.shape) == 2 or original_image.shape[2] == 1:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
    for image_number in range(len(circles)):
        x, y, r = circles[image_number]
        
        # กำหนดสีของวงกลมและข้อความตามเงื่อนไข
        if text_result[image_number].lower() == 'ok':
            circle_color = (0, 255, 0)  # สีเขียว
            text_color = (0, 255, 0)   # สีเขียว
        elif text_result[image_number].lower() == 'ng':
            circle_color = (0, 0, 255)  # สีแดง
            text_color = (0, 0, 255)   # สีแดง
        elif text_result[image_number].lower() == 'undefined':
            circle_color = (255, 0, 0)  # สีน้ำเงิน
            text_color = (255, 0, 0)   # สีน้ำเงิน
        else:
            # กรณีที่ข้อความไม่ตรงกับเงื่อนไขใด ๆ ให้ใช้ค่าเริ่มต้น
            circle_color = (255, 255, 255)  # สีขาว
            text_color = (255, 255, 255)   # สีขาว

        # วาดวงกลม
        cv2.circle(original_image, (x, y), r, circle_color, 5)

        # คำนวณตำแหน่งข้อความให้อยู่ตรงกลางด้านบนของวงกลม
        font_scale = 6
        font_thickness = 7
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text_result[image_number], font, font_scale, font_thickness)[0]
        text_x = x - text_size[0] // 2  # จัดข้อความให้อยู่ตรงกลางในแนวนอน
        text_y = y - r - 10  # วางข้อความด้านบนของวงกลม (ระยะห่าง 10 px)

        # ใส่ข้อความลงในภาพ
        cv2.putText(original_image, text_result[image_number], (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return original_image


def process_6_save_image(raw_image,result_image,roi_imag,predict_image):
    number_raw_folder = len([
        item for item in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, item)) and item.startswith("result_")
    ])
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # นับจำนวนไฟล์ในโฟลเดอร์
    file_count = len(os.listdir(folder_path)) + 1
    # สร้าง result_path
    result_folder = f"result_{file_count}_{timestamp}"
    result_path = os.path.join(folder_path, f"result_{file_count}_{timestamp}")
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    #save raw file
    for number_raw_file_name in range(1,len(raw_image)+1):
        raw_filename = os.path.join(result_path, "raw_image_{}.jpg".format(number_raw_file_name))
        cv2.imwrite(raw_filename, raw_image[number_raw_file_name-1])
        #print(f"save at: {raw_filename}")
    normal_map_visual_filename = os.path.join(result_path, "normal_map_visual_image.jpg")
    cv2.imwrite(normal_map_visual_filename, normal_map_visual)
    #print(f"save at: {normal_map_visual_filename}")
    result_image_filename = os.path.join(result_path, "result_image.jpg")
    cv2.imwrite(result_image_filename, result_image)
    for number_roi_file_name in range(1,len(roi_imag)+1):
        roi_filename = os.path.join(result_path, "roi_image_{}.jpg".format(number_roi_file_name))
        cv2.imwrite(roi_filename, roi_imag[number_roi_file_name-1])
        #print(f"save at: {roi_filename}")
    for number_result_file_name in range(1,len(predict_image)+1):
        result_filename = os.path.join(result_path, "predict_roi_{}.jpg".format(number_result_file_name))
        cv2.imwrite(result_filename, predict_image[number_result_file_name-1])
        #print(f"save at: {result_filename}")
    print(f"save at: {result_path}")
    return result_folder

def process_7_save_db(name, text_result):
    result_1, result_2 = text_result
    # ตั้งค่าการเชื่อมต่อฐานข้อมูล
    server = "192.168.0.203"  # แทนด้วย IP หรือชื่อเซิร์ฟเวอร์ของคุณ
    user = "sa"               # ชื่อผู้ใช้
    password = "sa$admin123" # รหัสผ่าน
    database = "oat_camera_db" # ชื่อฐานข้อมูล

    # ดึงวันที่และเวลาปัจจุบัน
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    try:
        # เชื่อมต่อกับฐานข้อมูล
        conn = pymssql.connect(server, user, password, database)
        cursor = conn.cursor()

        # คำสั่ง SQL สำหรับเพิ่มข้อมูล
        insert_query = """
        INSERT INTO camera_tb (name, date, time, result1, result2)
        VALUES (%s, %s, %s, %s, %s)
        """

        # เพิ่มข้อมูลลงในตาราง
        cursor.execute(insert_query, (name, current_date, current_time, result_1, result_2))

        # ยืนยันการเปลี่ยนแปลง
        conn.commit()
        print("Data inserted successfully.")

    except pymssql.Error as e:
        print("Error occurred:", e)

    finally:
        # ปิดการเชื่อมต่อ
        if conn:
            conn.close()

if __name__ == "__main__":
###############-----------------------------------------------get picture from camera and save to raw_folder_path--------------------------------------------------------#####################
    basler_camera = OpenBaslerCamera()
    basler_camera.start_grabbing()
    model_path = r"/home/mic/Camera_project/basler/version_4_save_final/weight_02.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ",device)
    model = YOLO(model_path).to(device)
    print("Program is started")
    print("press 't' to save or 'q' for exit")
    raw_image = []
    number_raw_file = 0
    while True:
        img = basler_camera.get_image()
        if img is None:
            print("cannot get picture from camera")
            break
        # Display the image
        cv2.imshow('original_img', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('t'):
            number_raw_file += 1
            print(f"capture picture {number_raw_file}")
            img_blurred = cv2.medianBlur(img, 5)  # การเบลอด้วย GaussianBlur
            gray_img = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)  # แปลงเป็นขาวดำ
            raw_image.append(gray_img)
            if number_raw_file % 4 == 0:
                start_time_process_all = time.time()
                print("Process 2 photometric stereo")
                start_time_process2 = time.time()
                normal_map_visual,B,G,R = process_2_photometric_stereo(raw_image)
                print(f"end process 2 with {time.time()-start_time_process2}")
                print("Process 3 cropROI")
                start_time_process3 = time.time()
                roi = process_3_crop_ROI(B)
                roi_show = roi.copy()
                print(f"end process 3 with {time.time()-start_time_process3}")
                start_time_process4 = time.time()
                predict_image,text_result = process_4_image_classification(roi_show)
                print(f"end process 4 with {time.time()-start_time_process4}")
                print("Process 5 draw result")
                start_time_process5 = time.time()
                result_image = process_5_draw_result(B,text_result)
                print(f"end process 5 with {time.time()-start_time_process5}")
                print("Process 6 saveImage")
                start_time_process6 = time.time()
                result_folder = process_6_save_image(raw_image,result_image,roi_show,predict_image)
                print(f"end process 6 with {time.time()-start_time_process6}")
                print("Process 7 saveDB")
                start_time_process7 = time.time()
                process_7_save_db(result_folder,text_result)
                print(f"end process 7 with {time.time()-start_time_process7}")
                
                
                print(f"end process all with {time.time()-start_time_process_all}")

                cv2.imshow('result_img', result_image)
                cv2.imshow('roi_1', roi[0])
                cv2.imshow('roi_2', roi[1])
                
                raw_image = []
                number_raw_file = 0
                print("press 't' to save or 'q' for exit")
            elif key == 27 or key == ord('q'):  # Exit if 'Esc' key is pressed
                break
    basler_camera.release()
    cv2.destroyAllWindows()

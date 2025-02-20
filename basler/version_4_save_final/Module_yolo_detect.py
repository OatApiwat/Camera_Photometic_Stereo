import os
import cv2
from ultralytics import YOLO
import datetime
import time
import torch
class YOLOImageClassifier:
    def __init__(self, model_path):
        """
        Constructor สำหรับโหลดโมเดล YOLO
        """
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("device: ",self.device)
        self.model = YOLO(model_path).to(self.device)

    def classify_images(self, data_path, output_path, original_image_path, circles):
        """
        ฟังก์ชันสำหรับทำนายผลลัพธ์และบันทึกภาพที่มีผลลัพธ์
        """
        # โหลดภาพต้นฉบับสำหรับวาดวงกลม
        original_image = cv2.imread(original_image_path)
        result = []  # สำหรับเก็บผลลัพธ์ของแต่ละภาพ

        # สร้างโฟลเดอร์สำหรับบันทึกผลลัพธ์ หากยังไม่มี
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # วนลูปผ่านไฟล์ภาพใน data_path
        
        image_number = 0
        for filename in os.listdir(data_path):
            
            if filename.endswith(".jpg") or filename.endswith(".png"):
                
                image_path = os.path.join(data_path, filename)
                img = cv2.imread(image_path)
                #img = cv2.resize(img, (640, 640)) 
                # ทำการทำนายภาพ
                results = self.model(image_path,save = False)
                # ดึงข้อมูลความมั่นใจจาก probs
                if results[0].probs is not None:
                    probs = results[0].probs.data.cpu().numpy()  # แปลง Tensor เป็น NumPy array
                    print(f"Results for {filename}:")
                    
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
                   
                    # เพิ่มข้อความผลลัพธ์ในภาพ
                    text = f"Result: {final_result}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img, text, (10, 30), font, 1, text_color, 2, cv2.LINE_AA)

                    # วาดวงกลมใน original_image
                    x, y, r = circles[image_number]
                    #print("image_number: ", image_number)
                    #print(" x, y, r: ", x, " : ", y, " : ", r)
                    cv2.circle(original_image, (x, y), r, circle_color, 5)

                    # คำนวณตำแหน่งข้อความให้อยู่ตรงกลางด้านบนของวงกลม
                    font_scale = 1
                    font_thickness = 2
                    text_size = cv2.getTextSize(final_result, font, font_scale, font_thickness)[0]
                    text_x = x - text_size[0] // 2  # จัดข้อความให้อยู่ตรงกลางในแนวนอน
                    text_y = y - r - 10  # วางข้อความด้านบนของวงกลม (ระยะห่าง 10 px)

                    # ใส่ข้อความลงในภาพ
                    cv2.putText(original_image, final_result, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    image_number += 1
                    # บันทึกภาพที่โฟลเดอร์ output_path
                    output_file = os.path.join(output_path, filename)
                    cv2.imwrite(output_file, img)
                    result.append(final_result)
                    
                    
                    
                    
        # บันทึก original_image พร้อมวงกลม
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        original_output_path = os.path.join(output_path, f"final_result_{current_time}.png")
        cv2.imwrite(original_output_path, original_image)
        

        return result,original_image

if __name__ == "__main__":
    # กำหนดเส้นทางโมเดลและโฟลเดอร์ข้อมูล
    
    model_path = r"/home/mic/Camera_project/basler/version_2/weight_02.pt"
    
    data_path = r"/home/mic/Camera_project/basler/version_1/ROI_1_test"
    output_path = r"/home/mic/Camera_project/basler/version_2/4_Result"
    original_image_path = r"/home/mic/Camera_project/basler/version_1/Result_1/B.png"
    circles = [
        (375, 1430, 170),  # วงกลมที่ 1
        (870, 545, 170),   # วงกลมที่ 2
    ]
    

    # สร้าง object ของ YOLOImageClassifier
    
    classifier = YOLOImageClassifier(model_path)
    

    # เรียกใช้ฟังก์ชัน classify_images
    
    result = classifier.classify_images(data_path, output_path, original_image_path, circles)
    
    print("Result:", result)


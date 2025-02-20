import os
import cv2
import numpy as np

class PhotometricStereo:
    def __init__(self,input_folder,output_folder):
        # กำหนดทิศทางแหล่งกำเนิดแสง (สมมติว่าเรารู้ค่าที่แน่นอน)
        self.direction = 0.5
        self.light_directions = np.array([
        [-self.direction,  self.direction, 1],  # image_2: แสงมาจากมุมบนซ้าย
        [-self.direction, -self.direction, 1],  # image_1: แสงมาจากมุมล่างซ้าย
        [ self.direction, -self.direction, 1],  # image_4: แสงมาจากมุมล่างขวา
        [ self.direction,  self.direction, 1],  # image_3: แสงมาจากมุมบนขวา
        ], dtype=np.float32)
        # กำหนด path ของ input และ output
        #self.input_folder = r"/home/mic/Camera_project/basler/Picture_1"
        #self.output_folder = r"/home/mic/Camera_project/basler/Result_1"
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
    def ProcessImages(self):
        # อ่านภาพทั้งหมด และแปลงเป็นขาวดำ
        image_files = [f for f in os.listdir(self.input_folder) if f.endswith(".jpg")]
        images = []
        for f in image_files:
            img = cv2.imread(os.path.join(self.input_folder, f))  # อ่านภาพแบบสี
            # ใช้ GaussianBlur หรือ medianBlur เพื่อลบ noise
            img_blurred = cv2.medianBlur(img, 3)  # การเบลอด้วย GaussianBlur
            gray_img = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)  # แปลงเป็นขาวดำ
            images.append(gray_img)
        # แปลงภาพทั้งหมดเป็น numpy array
        images = np.stack(images, axis=-1).astype(np.float32)
        # Normalization ของทิศทางแสง
        light_directions = self.light_directions
        light_directions /= np.linalg.norm(light_directions, axis=1, keepdims=True)
        # คำนวณ Normal Map
        height, width, num_images = images.shape
        intensity = images.reshape(-1, num_images)
        normal_map = np.dot(np.linalg.pinv(light_directions), intensity.T).T
        normal_map = normal_map.reshape(height, width, 3)

        # Normalize เพื่อให้ normal map เป็น [0, 1]
        normal_map_magnitude = np.linalg.norm(normal_map, axis=2, keepdims=True)
        normal_map /= np.maximum(normal_map_magnitude, 1e-5)
        normal_map_visual = ((normal_map + 1) / 2 * 255).astype(np.uint8)  # สำหรับแสดงผล

        # การทำ median filter เพื่อลด noise
        normal_map_visual = cv2.medianBlur(normal_map_visual, 5)  # ใช้ขนาด kernel 5x5
        
        (B, G, R) = cv2.split(normal_map_visual)
        #cv2.imwrite(os.path.join(self.output_folder, "normal_map_visual.png"), normal_map_visual)
        cv2.imwrite(os.path.join(self.output_folder, "B.png"), B)
        cv2.imwrite(os.path.join(self.output_folder, "G.png"), G)
        cv2.imwrite(os.path.join(self.output_folder, "R.png"), R)
        return normal_map_visual,B,G,R
if __name__ == "__main__":
    input_folder = r"/home/mic/Camera_project/basler/version_2/1_raw_folder/raw_picture_4"
    output_folder = r"/home/mic/Camera_project/basler/version_2/2_photometric_folder/test_time"
    processImage = PhotometricStereo(input_folder,output_folder)
    normal_map_visual,B,G,R = processImage.ProcessImages()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    window_width = 772  # ความกว้างของหน้าต่าง
    window_height = 516  # ความสูงของหน้าต่าง
    cv2.namedWindow('normal_map_visual',cv2.WINDOW_NORMAL)
    cv2.namedWindow('B',cv2.WINDOW_NORMAL)
    cv2.namedWindow('G',cv2.WINDOW_NORMAL)
    cv2.namedWindow('R',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('normal_map_visual', window_width, window_height)  # ปรับขนาดหน้าต่าง
    cv2.resizeWindow('B', window_width, window_height)  # ปรับขนาดหน้าต่าง
    cv2.resizeWindow('G', window_width, window_height)  # ปรับขนาดหน้าต่าง
    cv2.resizeWindow('R', window_width, window_height)  # ปรับขนาดหน้าต่าง
    cv2.imshow('normal_map_visual', normal_map_visual)
    cv2.imshow('B', B)
    cv2.imshow('G', G)
    cv2.imshow('R', R)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


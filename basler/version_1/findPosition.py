import cv2
import numpy as np
import os

class ImageProcessor:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.x = 870
        self.y = 545
        self.r = 170
        #self.x = 375
        #self.y = 1430
        #self.r = 170
        #self.x = 375
        #self.y = 1430
        #self.r = 170

    def nothing(self, val):
        pass

    def create_trackbar(self):
        # สร้างหน้าต่างเพื่อแสดง trackbars
        cv2.namedWindow('Trackbars')
        cv2.createTrackbar('X', 'Trackbars', self.x, 3000, self.nothing)  # ค่า max 1920
        cv2.createTrackbar('Y', 'Trackbars', self.y, 3000, self.nothing)  # ค่า max 1080
        cv2.createTrackbar('R', 'Trackbars', self.r, 500, self.nothing)  # ค่า max 500

    def update_trackbar_values(self):
        # อ่านค่า X, Y, R จาก trackbar
        self.x = cv2.getTrackbarPos('X', 'Trackbars')
        self.y = cv2.getTrackbarPos('Y', 'Trackbars')
        self.r = cv2.getTrackbarPos('R', 'Trackbars')

    def display_images_with_circle(self):
        # อ่านไฟล์ภาพจาก input_folder
        image_files = [f for f in os.listdir(self.input_folder) if f.endswith(('.jpg', '.png'))]

        for image_file in image_files:
            image_path = os.path.join(self.input_folder, image_file)
            image = cv2.imread(image_path)

            while True:
                # อัปเดตค่าของ trackbar
                self.update_trackbar_values()

                # วาดวงกลมบนภาพ
                image_copy = image.copy()  # สร้างสำเนาภาพเพื่อไม่ให้เกิดการเขียนทับ
                cv2.circle(image_copy, (self.x, self.y), self.r, (0, 255, 0), 2)  # วงกลมสีเขียว (BGR)

                # แสดงภาพที่มีวงกลม
                window_width = 772  # ความกว้างของหน้าต่าง
                window_height = 516  # ความสูงของหน้าต่าง
                cv2.namedWindow(f"Image: {image_file}",cv2.WINDOW_NORMAL)
                cv2.resizeWindow(f"Image: {image_file}", window_width, window_height)  # ปรับขนาดหน้าต่าง
                cv2.imshow(f"Image: {image_file}", image_copy)

                # รอให้กดปุ่มเพื่อปิดภาพ
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # กด 'q' เพื่อออกจาก loop
                    break

    def run(self):
        # เริ่มต้น trackbars
        self.create_trackbar()

        while True:
            # แสดงภาพทั้งหมดใน folder พร้อมวงกลม
            self.display_images_with_circle()

            # แสดง trackbars
            
            cv2.imshow('Trackbars', np.zeros((300, 300)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # กด 'q' เพื่อออกจาก loop
                break

        cv2.destroyAllWindows()

# ใช้ if __name__ == "__main__" เพื่อให้โปรแกรมทำงานได้เมื่อต้องการ
if __name__ == "__main__":
    input_folder = r"/home/mic/Camera_project/basler/version_1/Result_2"

    # สร้างออบเจ็กต์และรันโปรแกรม
    processor = ImageProcessor(input_folder)
    processor.run()


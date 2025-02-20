from open_basler_camera import OpenBaslerCamera
import cv2
import os

cv2.namedWindow('original_img',cv2.WINDOW_NORMAL)
window_width = 772  # ความกว้างของหน้าต่าง
window_height = 516  # ความสูงของหน้าต่าง
cv2.resizeWindow('original_img', window_width, window_height)  # ปรับขนาดหน้าต่าง

# กำหนดเส้นทางสำหรับบันทึกรูปภาพ
save_path = r"/home/mic/Camera_project/basler/Picture_4"
# ตรวจสอบว่าโฟลเดอร์ปลายทางมีอยู่หรือไม่ ถ้าไม่มีให้สร้างขึ้น
if not os.path.exists(save_path):
    os.makedirs(save_path)

if __name__ == "__main__":
    basler_camera = OpenBaslerCamera()
    basler_camera.start_grabbing()
    print("press 't' to save or 'q' for exit")
    while True:
        img = basler_camera.get_image()
        if img is None:
            print("cannot get picture from camera")
            break
        # Display the image
        cv2.imshow('original_img', img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('t'):
            # สร้างชื่อไฟล์ตามเวลาปัจจุบัน
            filename = os.path.join(save_path, "image_{}.jpg".format(len(os.listdir(save_path)) + 1))
            cv2.imwrite(filename, img)
            print(f"save at: {filename}")
        elif key == 27 or key == ord('q'):  # Exit if 'Esc' key is pressed
            break

    # Release resources
    basler_camera.release()
    cv2.destroyAllWindows()

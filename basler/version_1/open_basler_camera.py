from pypylon import pylon
import cv2

class OpenBaslerCamera:
    def __init__(self):
        # Connecting to the first available camera
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

        # Setting up the image converter for OpenCV compatibility
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def start_grabbing(self):
        # Start grabbing video continuously
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def stop_grabbing(self):
        # Stop grabbing
        self.camera.StopGrabbing()

    def get_image(self):
        # Grab a frame from the camera
        if self.camera.IsGrabbing():
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                # Convert to OpenCV-compatible format
                image = self.converter.Convert(grabResult)
                img = image.GetArray()
                
                return img
            else:
                print("Failed to grab image.")
                return None
            grabResult.Release()
        return None

    def release(self):
        # Release resources when done
        self.stop_grabbing()
        cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    basler_camera = OpenBaslerCamera()
    basler_camera.start_grabbing()

    while True:
        img = basler_camera.get_image()
        if img is not None:
            # Display the image
            cv2.imshow('Basler Camera Image', img)

        k = cv2.waitKey(1)
        if k == 27:  # Exit if 'Esc' key is pressed
            break

    # Release resources
    basler_camera.release()


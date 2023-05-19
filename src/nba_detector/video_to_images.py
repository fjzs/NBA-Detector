import cv2
import os
import random

def generate_sample_frames(folder_path: str, file_extension:str, number_frames:int):
    """
    Given a folder_path and a file_extension I generate a random set of number_frames images
    of the videos in that folder. Use this website (for example) to download the video
    https://ssyoutube.com/en79/youtube-video-downloader
    :param folder_path:
    :param file_extension:
    :param number_frames:
    :return:
    """

    filepath = os.path.join(folder_path, file_extension)
    if not os.path.exists(filepath):
        raise ValueError("filepath does not exist")

    # Create the new folder if it does not exist
    file = file_extension.split(".")[0]  # ex: vid01
    folder = os.path.join(folder_path, file)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Load the video
    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check if camera opened successfully
    if not cap.isOpened():
        raise ValueError("Error opening video stream or file")

    # Get a random list of frames to create
    random_frames = list(range(total_frames))
    random.shuffle(random_frames)
    random_frames = random_frames[0:number_frames]

    # Read specific frames
    ZERO_PAD = 5
    for i in random_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        image_path = os.path.join(folder, file + "_" + str(i).zfill(ZERO_PAD) + ".jpg")
        cv2.imwrite(image_path, frame)
        print(f"Created {image_path}")

    # When everything done, release the video capture object and close the frames
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    folder = "..\\NBA-videos"
    files = ["v10"]
    frames_per_video = 50
    for f in files:
        file_and_extension = f + ".mp4"
        generate_sample_frames(folder, file_and_extension, frames_per_video)


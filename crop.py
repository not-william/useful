import cv2
import numpy as np
import argparse
from pathlib import Path

def crop_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image needs to be numpy array")

    if len(image.shape) == 2:
        alpha = image
    elif len(image.shape) == 3:
        alpha = image[:, :, -1]
    else:
        raise ValueError("image must have 1 or 3 channels")

    x1 = alpha.max(axis=1).argmax()
    x2 = alpha.shape[1] - alpha.max(axis=1)[::-1].argmax()
    y1 = alpha.max(axis=0).argmax()
    y2 = alpha.shape[0] - alpha.max(axis=0)[::-1].argmax()


    if len(image.shape) == 2:
        image_cropped = image[x1:x2, y1:y2]
    else:
        image_cropped = image[x1:x2, y1:y2, :]

    return image_cropped

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop an image')
    parser.add_argument("input", type=str, help="input image", nargs="+")
    parser.add_argument("--output_directory", dest="output_dir", help="optional")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    for file in args.input:
        file = Path(file)
        if not file.is_file():
            raise ValueError(f"{args['input']} is not a file")

        image = cv2.imread(file.as_posix(), -1)

        cropped_image = crop_image(image)

        if args.output_dir:
            output_file = output_dir / file.name
        else:
            output_file = file.parent / (file.stem + "_cropped" + file.suffix)

        cv2.imwrite(output_file.as_posix(), cropped_image)


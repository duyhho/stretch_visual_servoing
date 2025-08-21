import cv2
import os
import re # Used for reading numbers from filenames

def create_panoramas(input_dir="scan_images_async"):
    """
    Loads images from the scan, groups them by tilt angle,
    and stitches each group into a panoramic image.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Directory '{input_dir}' not found.")
        return

    print("Organizing images...")
    images_by_tilt = {}

    # Regex to find the pan and tilt values in the filename
    # e.g., "scan_pan_-229_tilt_-105.jpg"
    pattern = re.compile(r"scan_tilt_(-?\d+)_pan_(-?\d+)\.jpg")

    for filename in os.listdir(input_dir):
        match = pattern.match(filename)
        if match:
            pan_deg = int(match.group(1))
            tilt_deg = int(match.group(2))

            # Group filenames by their tilt angle
            if tilt_deg not in images_by_tilt:
                images_by_tilt[tilt_deg] = []
            images_by_tilt[tilt_deg].append((pan_deg, os.path.join(input_dir, filename)))

    if not images_by_tilt:
        print("No images found with the expected filename format.")
        return

    # Create a stitcher object
    stitcher = cv2.Stitcher_create()

    print(f"\nFound {len(images_by_tilt)} tilt levels to process.")

    # Process each tilt level separately
    for tilt_deg in sorted(images_by_tilt.keys()):
        print(f"--- Processing Tilt Angle: {tilt_deg} degrees ---")

        # Sort the images for this tilt level by their pan angle (left to right)
        image_paths = sorted(images_by_tilt[tilt_deg])

        # Load the images into a list
        images = [cv2.imread(path) for pan, path in image_paths]

        if len(images) < 2:
            print("  Not enough images to stitch for this level. Skipping.")
            continue

        print(f"  Stitching {len(images)} images...")
        (status, stitched_image) = stitcher.stitch(images)

        if status == cv2.Stitcher_OK:
            # Create a filename for the output panorama
            output_filename = f"panorama_tilt_{tilt_deg}.jpg"
            cv2.imwrite(output_filename, stitched_image)
            print(f"  Success! Panorama saved to '{output_filename}'")
        else:
            print(f"  Error: Stitching failed with status code {status}. Try adjusting scan overlap.")

    print("\nAll panoramas created.")

if __name__ == '__main__':
    create_panoramas()
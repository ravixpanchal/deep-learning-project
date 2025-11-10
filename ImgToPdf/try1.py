import os
from pdf2image import convert_from_path
from PIL import Image

def crop_image(image, crop_ratios):
    """
    Crop image according to ratios (top, right, bottom, left)
    Ratios should be between 0 and 1 (fraction of the dimension)
    """
    width, height = image.size
    top = int(height * crop_ratios[0])
    right = int(width * (1 - crop_ratios[1]))
    bottom = int(height * (1 - crop_ratios[2]))
    left = int(width * crop_ratios[3])
    return image.crop((left, top, right, bottom))

def convert_pdf_in_folder(folder_path, crop_ratios=(0, 0, 0, 0), output_folder="RawOutput_jpg3", dpi=300):
    """
    Converts all PDFs in a folder to cropped JPGs.
    crop_ratios: tuple(top, right, bottom, left) as fractions (0–1)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print("No PDF files found in the folder.")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        base_name = os.path.splitext(pdf_file)[0]
        print(f"Processing: {pdf_file} ...")

        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            for i, img in enumerate(images):
                cropped_img = crop_image(img, crop_ratios)
                output_filename = f"{base_name}_page_{i+1}.jpg"
                output_path = os.path.join(output_folder, output_filename)
                cropped_img.save(output_path, "JPEG", quality=95)
            print(f"✅ Converted and cropped {pdf_file}")
        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {e}")

if __name__ == "__main__":
    # Example usage:
    folder_path = input("Enter the folder path containing PDFs: ").strip()
    # Example crop ratios: (top, right, bottom, left)
    crop_ratios = (0.178, 0.00, 0.206, 0.00)
    convert_pdf_in_folder(folder_path, crop_ratios)

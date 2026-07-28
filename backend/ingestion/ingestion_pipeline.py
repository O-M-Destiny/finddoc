import os
import pickle
import pdfplumber

from unstructured.partition.pdf import partition_pdf



CACHE_DIR = os.path.join(os.path.dirname(__file__), "partitioned_dataset")
os.makedirs(CACHE_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(__file__)
pdf_path = os.path.join(BASE_DIR, 'pdf', 'Nvidia-Report25.pdf')

class Ingestion:
    def __init__(self, pdf_file_path: str):
        self.pdf_file_path = pdf_file_path

    def partition_document(self):
        base_name = os.path.splitext(os.path.basename(self.pdf_file_path))[0]
        cache_file = os.path.join(CACHE_DIR, f"{base_name}_elements.pkl")
        
        if os.path.exists(cache_file):
            print(f'Loading elements from Cache: {cache_file}')
            with open(cache_file, 'rb') as f:
                elements = pickle.load(f)

        else:
            print(f'Partitioning File: {self.pdf_file_path}')
            elements = partition_pdf(
                filename=self.pdf_file_path,
                strategy='hi_res',
                infer_table_structure=True,
                extract_image_block_to_payload=True,
                extract_image_block_types=["Image"]
                )
            with open(cache_file, 'wb') as f:
                pickle.dump(elements, f)

        print(f"Returned Number of Elements: {len(elements)}")
        return elements
    # how many datatypes it contains, # number of elements #filer elements
    def _document_info(self):
        base_name = os.path.splitext(os.path.basename(self.pdf_file_path))[0]
        cache_file = os.path.join(CACHE_DIR, f"{base_name}_elements.pkl")
        # NO of element Info
        try:
            with open(cache_file, 'rb') as f:
                partitioned_elements = pickle.load(f)
                
        except Exception as e:
            print(f'Error: {e}')

        # Available Datatype Info
        available_datatypes = set([str(type(el))for el in partitioned_elements])

        print(f'Number of Partitioned Elements: {len(partitioned_elements)}')
        print()
        print(f'Available Number of Datatypes contained in Element')
        for i, dt in enumerate(available_datatypes, start=1):
            print(f'{i}. {dt}')

    

class DocumentCleaner:
    def __init__(self, elements: list, pdf_file_path: str):
        self.elements = elements
        self.pdf_file_path = pdf_file_path
    
    def inject_pdfplumber(self, padding_x=25):
        table_elements = [el for el in self.elements if el.category == "Table"]
        print(f"Injecting pdfplumber into {len(table_elements)} table elements...")

        with pdfplumber.open(self.pdf_file_path) as pdf:
            for element in table_elements:
                has_to_dict = hasattr(element, "to_dict")
                el_dict = element.to_dict() if has_to_dict else vars(element)

                metadata = el_dict.get("metadata", {})
                coordinates = metadata.get("coordinates", {})
                page_num = metadata.get("page_number")
                points = coordinates.get("points")

                if not page_num or not points:
                    continue

                try:
                    page = pdf.pages[page_num - 1]

                    base_x0 = float(min(p[0] for p in points))
                    base_y0 = float(min(p[1] for p in points))
                    base_x1 = float(max(p[0] for p in points))
                    base_y1 = float(max(p[1] for p in points))

                    layout_w = float(coordinates.get("layout_width", 2888))
                    layout_h = float(coordinates.get("layout_height", 3763))

                    raw_x0 = max(0.0, base_x0 - padding_x)
                    raw_y0 = base_y0
                    raw_x1 = min(layout_w, base_x1 + padding_x)
                    raw_y1 = base_y1

                    if layout_w <= 1.0 or layout_h <= 1.0:
                        x0 = (raw_x0 / max(raw_x1, 2000.0)) * float(page.width)
                        y0 = (raw_y0 / max(raw_y1, 2000.0)) * float(page.height)
                        x1 = (raw_x1 / max(raw_x1, 2000.0)) * float(page.width)
                        y1 = (raw_y1 / max(raw_y1, 2000.0)) * float(page.height)
                    else:
                        x0 = (raw_x0 / layout_w) * float(page.width)
                        y0 = (raw_y0 / layout_h) * float(page.height)
                        x1 = (raw_x1 / layout_w) * float(page.width)
                        y1 = (raw_y1 / layout_h) * float(page.height)

                    x0 = max(0.0, min(x0, float(page.width) - 1.0))
                    y0 = max(0.0, min(y0, float(page.height) - 1.0))
                    x1 = max(x0 + 1.0, min(x1, float(page.width)))
                    y1 = max(y0 + 1.0, min(y1, float(page.height)))

                    cropped_page = page.crop((x0, y0, x1, y1))

                    table_settings = {
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_tolerance": 3,
                        "join_tolerance": 3
                    }

                    extracted_tables = cropped_page.extract_tables(table_settings=table_settings)

                    final_content = []
                    is_structured = False

                    if extracted_tables:
                        is_structured = True
                        for table in extracted_tables:
                            table_rows = []
                            for row in table:
                                clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
                                if any(clean_row):
                                    table_rows.append(" | ".join(clean_row))
                            if table_rows:
                                final_content.append("\n".join(table_rows))
                    else:
                        raw_bounded_text = cropped_page.extract_text()
                        if raw_bounded_text and raw_bounded_text.strip():
                            final_content.append(raw_bounded_text.strip())
                        else:
                            broad_crop = page.crop((0, y0, float(page.width), y1))
                            broad_text = broad_crop.extract_text()
                            if broad_text and broad_text.strip():
                                final_content.append(broad_text.strip())

                    extracted_output_string = "\n\n---\n\n".join(final_content) if final_content else " No content retrieved."

                    element.metadata.pdfplumber_extracted = extracted_output_string
                    element.metadata.pdfplumber_is_table = is_structured

                except Exception as e:
                    element.metadata.pdfplumber_extracted = f"❌ Processing error: {str(e)}"
                    element.metadata.pdfplumber_is_table = False

        print("pdfplumber injection complete")
        return self

    # def remove_junk_tables(self):
        before = len(self.elements)

        def is_junk(el):
            if el.category != "Table":
                return False
            empty_html = el.metadata.text_as_html.count("<td>") == el.metadata.text_as_html.count("<td></td>")
            no_plumber = getattr(el.metadata, "pdfplumber_extracted", "") in ["❌ No content retrieved.", ""]
            return empty_html and no_plumber

        self.elements = [el for el in self.elements if not is_junk(el)]
        print(f"Removed junk tables: {before - len(self.elements)} elements dropped")
        return self
    
    def remove_junk_images(self):
        before = len(self.elements)
        junk_image_pages = [7, 14, 1, 10, 127]

        def is_junk(el):
            if el.category != "Image":
                return False
            if el.metadata.detection_class_prob <= 0.4:
                return True
            if len(el.text.strip()) <= 20:
                return True
            if el.metadata.page_number in junk_image_pages:
                return True
            return False

        self.elements = [el for el in self.elements if not is_junk(el)]
        print(f"Removed junk images: {before - len(self.elements)} elements dropped")
        return self
    
    def clean(self):
        return (self
                .inject_pdfplumber()
                .remove_junk_images()
                # .remove_junk_tables()
                )

    def get_elements(self):
        return self.elements
    
    def save(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(self.elements, f)
        return self


ingestion = Ingestion(pdf_file_path=pdf_path)
elements = ingestion.partition_document()

cleaned_cache = os.path.join(CACHE_DIR, "Nvidia-Report25_cleaned.pkl")

if os.path.exists(cleaned_cache):
    with open(cleaned_cache, 'rb') as f:
        cleaned_elements = pickle.load(f)
    print(f"Loaded {len(cleaned_elements)} cleaned elements")
else:
    cleaner = DocumentCleaner(elements=elements, pdf_file_path=pdf_path)
    cleaned_elements = cleaner.clean().save(cleaned_cache).get_elements()

print(f"Final element count: {len(cleaned_elements)}")
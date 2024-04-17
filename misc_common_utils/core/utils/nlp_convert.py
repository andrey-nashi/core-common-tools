import fitz


def nlp_convert_pdf2text(path_pdf: str) -> list:
    text_on_page = []

    f_descriptor = fitz.open(path_pdf)

    for page_index, page in enumerate(f_descriptor):
        text = page.get_text()
        text_on_page.append(text)

    f_descriptor.close()
    return text_on_page

